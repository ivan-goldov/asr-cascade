import logging
import random
import sys
from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from common.decoder import Decoder
from common.dictionary import Dictionary
from common.objective import CrossEntropyLoss
from common.progress_tracker import ProgressTracker
from common.train_utils import all_reduce, nccl_barrier_on_cpu, ProgressListener, TorchOptimizerWrapper
from common.utils import gpu_id, num_gpus, Timer
from common.yt_utils import BatchBlock

from lm.yt_dataset import TextBatch, YtTestDataset

LOG = logging.getLogger()


class MetricCalcer(ProgressListener):
    def __init__(self, decoder: Decoder, summary_writer: SummaryWriter):
        self._decoder = decoder
        self._summary_writer = summary_writer

    def on_train_freq(self, progress_tracker: ProgressTracker, batch: TextBatch, loss: Tensor, output: Tensor,
                      **kwargs):
        loss = loss.item()
        ppl = kwargs["perplexity"]

        LOG.info(f"Epoch: {progress_tracker.epoch()}, Iteration: {progress_tracker.iteration()}")
        LOG.info(f"Loss = {loss}, PPL = {ppl.item()}")

        if self._summary_writer is not None and gpu_id() == 0:
            self._summary_writer.add_scalar("Loss/train", loss, progress_tracker.total_iterations())
            self._summary_writer.add_scalar("PPL/train", ppl, progress_tracker.total_iterations())


class TestMetricCalcer(ProgressListener):

    def __init__(self, test_dataset: Optional[YtTestDataset], model: Module, objective: Module, decoder: Decoder,
                 summary_writer: SummaryWriter):
        self._model = model
        self._objective = objective
        self._decoder = decoder
        self._test_dataset = test_dataset
        self._summary_writer = summary_writer

    def on_train_freq(self, progress_tracker: ProgressTracker, batch: TextBatch, loss_value: Tensor,
                      log_probs: Tensor, **kwargs):
        pass

    def on_test_freq(self, progress_tracker: ProgressTracker):
        if self._test_dataset is None:
            return
        self._model.eval()
        timer = Timer()
        sys.stderr.write("Will compute scores\n")
        with torch.no_grad():
            num_batches = 0
            loss = log_likelihood = num_words = 0
            for batch in self._test_dataset:
                output = self._model(batch.source_tokens, batch.tokens_lengths)
                batch_loss = compute_loss(self._objective, output, batch.target_tokens)
                batch_log_likelihood, batch_num_words = compute_perplexity(output, batch.target_tokens,
                                                                           self._decoder.dictionary(), reduce=False)
                loss += batch_loss.item()
                log_likelihood += batch_log_likelihood.item()
                num_words += batch_num_words.item()

                num_batches += 1

            sys.stderr.write(f"Evaluation batches on worker {gpu_id()}: {num_batches}\n")

            loss, log_likelihood, num_words = map(all_reduce, (loss, log_likelihood, num_words))
            loss /= self._test_dataset.total_samples
            ppl = torch.exp(torch.tensor((1 / num_words) * log_likelihood)).item()

            LOG.info(f"==========>>>>>>Evaluation Loss:    {loss}")
            LOG.info(f"==========>>>>>>Evaluation PPL:     {ppl}")

            if self._summary_writer is not None and gpu_id() == 0:
                self._summary_writer.add_scalar("Loss/test", loss, progress_tracker.total_iterations())
                self._summary_writer.add_scalar("PPL/test", ppl, progress_tracker.total_iterations())

        LOG.info(f"Evaluation time: {timer.passed()} seconds")


class ModelTrainer:
    def __init__(self, model: Module, objective: Module, optimizer: TorchOptimizerWrapper,
                 scheduler, progress_tracker: ProgressTracker, progress_listener: ProgressListener):
        self._model = model
        self._objective = objective
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._progress_tracker = progress_tracker
        self._progress_listener = progress_listener

    def epoch(self, data_loader: DataLoader, dictionary: Dictionary, num_iterations: int, batch_log_frequency: int,
              test_log_frequency: int):
        self._model.train()

        if self._progress_tracker.is_current_epoch_finished():
            self._progress_tracker.start_epoch()
            LOG.info(f"Training epoch {self._progress_tracker.epoch()}")
        else:
            LOG.info(f"Will continue epoch: {self._progress_tracker.epoch()}")

        debug_cursor = 0
        for batch_block in data_loader:
            assert len(batch_block) == 1
            batch_block: BatchBlock = batch_block[0]

            block_audio_length = 0

            # don't go from small to big samples always
            # seed is same on all GPUs, so speed should be balanced
            self._set_shuffle_seed(self._progress_tracker.iteration())
            random.shuffle(batch_block.batches)

            if num_gpus() > 1:
                torch.cuda.synchronize(torch.distributed.get_rank())
                nccl_barrier_on_cpu()

            for batch in batch_block:
                self._model.train()
                batch = batch.cuda()
                sys.stderr.write(f"Batch on GPU #{gpu_id()}\n")

                output = self._model(batch.source_tokens, batch.tokens_lengths)
                loss = compute_loss(self._objective, output, batch.target_tokens)
                loss /= len(batch)

                self._optimizer(loss)

                current_step = self._progress_tracker.iteration()
                if current_step % batch_log_frequency == 0:
                    sys.stderr.write(f"Debug cursor on GPU #{gpu_id()} is {debug_cursor}\n")
                    ppl = compute_perplexity(output, batch.target_tokens, dictionary, reduce=True)
                    self._progress_listener.on_train_freq(progress_tracker=self._progress_tracker,
                                                          batch=batch,
                                                          loss=loss,
                                                          output=output,
                                                          perplexity=ppl)
                if current_step % test_log_frequency == 0:
                    self._progress_listener.on_test_freq(progress_tracker=self._progress_tracker)

                self._progress_tracker.finish_batch()
                sys.stderr.write(f"Before finish batch listeners #{gpu_id()}\n")
                self._progress_listener.after_finish_batch()
                sys.stderr.write(f"After finish batch listeners #{gpu_id()}\n")

                debug_cursor += 1

                if self._scheduler is not None:
                    self._scheduler.step()

            sys.stderr.write(f"Finish read block on GPU #{gpu_id()} is {debug_cursor}\n")
            self._progress_tracker.finish_read_block(batch_block.read_thread_state,
                                                     all_reduce(block_audio_length))
            self._progress_listener.after_finish_batch_block()
            sys.stderr.write(f"Listeners were called on GPU #{gpu_id()}\n")

            if self._progress_tracker.total_iterations() >= num_iterations:
                self._progress_listener.on_test_freq(progress_tracker=self._progress_tracker)
                sys.stderr.write(f"Epoch finished on GPU #{gpu_id()}, total batches {debug_cursor}\n")
                # i"m lazy, but this should be moved to listeners
                self._progress_tracker.finish_epoch()
                self._progress_listener.after_finish_epoch()
                return

            if num_gpus() > 1:
                torch.cuda.synchronize(torch.distributed.get_rank())
                nccl_barrier_on_cpu()

        raise RuntimeError("train iterator should be infinite")

    def _set_shuffle_seed(self, iter):
        random.seed(iter)
        random.seed(random.randint(0, 10000000))


def compute_loss(objective, output, targets):
    if isinstance(objective, CrossEntropyLoss):
        loss = objective(logits=output, targets=targets)
    else:
        raise Exception(f"Unknown objective function: {objective}")

    return loss


def compute_perplexity(output: Tensor, targets: Tensor, dictionary: Dictionary, reduce: bool):
    num_words = 0
    for ids in targets.tolist():
        text = dictionary.decode(ids).strip()
        num_words += len(text.split(" "))
    num_words = torch.tensor(num_words).cuda()

    pad_id = dictionary.pad_id()
    log_likelihood = F.cross_entropy(output.view(-1, output.size(-1)), targets.view(-1), reduction="sum",
                                     ignore_index=pad_id)

    if reduce:
        return torch.exp((1 / num_words.float()) * log_likelihood)
    else:
        return log_likelihood, num_words
