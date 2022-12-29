import logging
import random
import sys
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.cuda import amp
from torch.nn import Module
from torch.utils.data import DataLoader, IterableDataset

from asr.helpers import monitor_asr_train_progress, process_evaluation_batch, process_evaluation_epoch
from asr.models import ModelResult
from asr.progress_tracker import ASRProgressTracker
from asr.disk_dataset import SpeechBatch
from asr.metrics import speaker_independent_word_error_rate

from common.decoder import Decoder
from common.dictionary import Dictionary
from common.objective.cross_entropy import CrossEntropyLoss
from common.objective.ctc import CTCLossNM
from common.train_utils import all_reduce, nccl_barrier_on_cpu, ProgressListener, TorchOptimizerWrapper
from common.utils import gpu_id, num_gpus, Timer, get_padding_mask
from common.disk_utils import BatchBlock

from evaluation.evaluation_method.evaluation_method import EvaluationMethod

from tqdm.notebook import tqdm

import wandb

LOG = logging.getLogger()


class MetricCalcer(ProgressListener):
    def __init__(self, decoder: Decoder):
        self._decoder = decoder
        self._time_to_save = False

    def on_train_freq(self, progress_tracker: ASRProgressTracker, batch: SpeechBatch, loss: Tensor,
                      outputs: List[Tensor]):
        loss = loss.item()
        predictions = [self._decoder.decode_probs(outp) for outp in outputs]
        train_wer = monitor_asr_train_progress(predictions, batch.tokens, batch.tokens_lengths,
                                               self._decoder)
        LOG.info(f"Epoch: {progress_tracker.epoch()}, Iteration: {progress_tracker.iteration()}")
        LOG.info(f"Loss = {loss}, WER = {train_wer}")
        wandb.log({"loss": loss, "train_wer": train_wer})

        if progress_tracker is not None and gpu_id() == 0:
            progress_tracker.add_scalar("Loss/train", loss)
            progress_tracker.add_scalar("WER/train", train_wer)


class TestMetricCalcer(ProgressListener):
    def __init__(self, test_datasets: Dict[str, Tuple[IterableDataset, List[EvaluationMethod]]],
                 model: Module, objective,
                 decoder: Decoder, dictionary, test_sz: int):
        self._model = model
        self._objective = objective
        self._decoder = decoder
        self._dictionary = dictionary
        datasets = {name: (list(test_dataset.__iter__(max_sz=test_sz)), m)
                    for name, (test_dataset, m) in test_datasets.items()}
        self.datasets = {name: ([
                                    b.cuda() if torch.cuda.is_available() else b
                                    for s in batches for b in s], m) for name, (batches, m) in datasets.items()}
        self._test_sz = test_sz

    def on_test_freq(self, progress_tracker: ASRProgressTracker):
        self._model.eval()
        timer = Timer()

        sys.stderr.write("Will compute scores\n")

        with torch.no_grad():
            for name, (batches, evaluation_methods) in self.datasets.items():
                for evaluation_method in evaluation_methods:
                    LOG.info(f"Starting {evaluation_method.name()} evaluation for {name} dataset")
                    references, hypotheses = [], []
                    for batch in batches:
                        hyps, _, refs = evaluation_method.evaluate(self._model, self._dictionary, batch)
                        references += refs
                        hypotheses += hyps
                    wer, _, _, wer_data = speaker_independent_word_error_rate(hypotheses, references, return_data=True)
                    LOG.info(f"==========>>>>>>{name} {evaluation_method.name()} evaluation WER:  {wer}")
                    for i in range(0, len(wer_data), 2):
                        for s in range(2):
                            diff_hyp, diff_ref, cur_wer = wer_data[i + s]
                            LOG.info(f'\tHyp{s + 1}: {str(diff_hyp)}')
                            LOG.info(f'\tRef{s + 1}: {str(diff_ref)}')
                            LOG.info(f'\tWER{s + 1}: {str(cur_wer)}')
                            LOG.info('')
                    wandb.log({f"total_wer_{name}_{evaluation_method.name()}": wer})

                LOG.info(f"Starting teacher_forced evaluation for {name} dataset")

                # tabbed backward
                _global_var_dict = {
                    "loss": 0,
                    "predictions": [],
                    "transcripts": [],
                }

                losses = []
                for batch in batches:
                    result: ModelResult = self._model(batch)
                    loss = compute_loss(self._objective, result, batch, self._decoder.dictionary())
                    loss /= len(batch)

                    predictions = [self._decoder.decode_probs(outp) for outp in result.output]
                    process_evaluation_batch(loss, predictions, batch.tokens, batch.tokens_lengths,
                                             _global_var_dict, self._decoder)
                    losses.append(loss.item())

                sys.stderr.write(f"Evaluation batches on worker {gpu_id()}\n")

                _global_var_dict['loss'] = sum(losses) / len(losses) if len(losses) != 0 else 0
                total_loss, total_wer, total_bleu = process_evaluation_epoch(_global_var_dict)

                LOG.info(f"==========>>>>>>{name} evaluation Loss: {total_loss}")
                LOG.info(f"==========>>>>>>{name} evaluation WER:  {total_wer}")
                wandb.log({f"total_loss_{name}": total_loss, f"total_wer_{name}": total_wer})

                if progress_tracker is not None and gpu_id() == 0:
                    metric_name = 'test' + ('_' if name != '' else '') + name
                    progress_tracker.add_scalar(f"Loss/{metric_name}", total_loss)
                    progress_tracker.add_scalar(f"WER/{metric_name}", total_wer)
                # tabbed backward

            LOG.info(f"Evaluation time: {timer.passed()} seconds")


class ModelTrainer:
    def __init__(self, model: Module, objective: Module, optimizer: TorchOptimizerWrapper,
                 progress_tracker: ASRProgressTracker, progress_listener: ProgressListener):
        self._model = model
        self._objective = objective
        self._optimizer = optimizer
        self._progress_tracker = progress_tracker
        self._progress_listener = progress_listener

    def epoch(self, data_loader: DataLoader, dictionary: Dictionary, epoch_duration_limit_seconds,
              batch_log_frequency: int, test_log_frequency: int):
        self._model.train()

        if self._progress_tracker.is_current_epoch_finished():
            self._progress_tracker.start_epoch()
            print(f"Training epoch {self._progress_tracker.epoch()}")
        else:
            print(f"Will continue epoch: {self._progress_tracker.epoch()}")

        print(f"Data loader: {type(data_loader)}")

        debug_cursor = 0
        for speech_batch_block in tqdm(data_loader):
            print("Started loading data")
            assert len(speech_batch_block) == 1
            speech_batch_block: BatchBlock = speech_batch_block[0]

            print("Speech batch block loaded")

            block_audio_length = 0

            # don't go from small to big samples always
            # seed is same on all GPUs, so speed should be balanced
            self._set_shuffle_seed(self._progress_tracker.iteration())
            random.shuffle(speech_batch_block.batches)

            if num_gpus() > 1:
                torch.cuda.synchronize(torch.distributed.get_rank())
                nccl_barrier_on_cpu()

            for batch in speech_batch_block:
                if debug_cursor == 0:
                    LOG.debug(f"Train id={gpu_id()} batch={batch.features}")
                self._model.train()
                if torch.cuda.is_available():
                    batch = batch.cuda()
                sys.stderr.write(f"Batch on GPU #{gpu_id()} shape={batch.features.shape}\n")

                with amp.autocast():
                    result: ModelResult = self._model(batch)
                    loss = compute_loss(self._objective, result, batch, dictionary)
                    loss /= len(batch)

                wandb.log({'batch_loss': loss})

                self._optimizer(loss)

                current_step = self._progress_tracker.iteration() + 1
                if current_step % batch_log_frequency == 0:
                    sys.stderr.write(f"Debug cursor on GPU #{gpu_id()} is {debug_cursor}\n")
                    self._progress_listener.on_train_freq(progress_tracker=self._progress_tracker,
                                                          batch=batch,
                                                          loss=loss,
                                                          output=result.output)
                if current_step % test_log_frequency == 0:
                    self._progress_listener.on_test_freq(progress_tracker=self._progress_tracker)

                self._progress_tracker.finish_batch()
                sys.stderr.write(f"Before finish batch listeners #{gpu_id()}\n")
                self._progress_listener.after_finish_batch()
                sys.stderr.write(f"After finish batch listeners #{gpu_id()}\n")

                debug_cursor += 1
                block_audio_length += batch.total_seconds

            sys.stderr.write(f"Finish read block on GPU #{gpu_id()} is {debug_cursor}\n")
            self._progress_tracker.finish_read_block(speech_batch_block.read_thread_state,
                                                     all_reduce(block_audio_length))
            sys.stderr.write(f"After finish_read_block()\n")
            self._progress_listener.after_finish_batch_block()
            sys.stderr.write(f"Listeners were called on GPU #{gpu_id()}\n")

            if self._progress_tracker.epoch_duration() > epoch_duration_limit_seconds:
                self._progress_listener.on_test_freq(progress_tracker=self._progress_tracker)
                sys.stderr.write(f"Epoch finished on GPU #{gpu_id()}, total batches {debug_cursor}\n")
                # i"m lazy, but this should be moved to listeners
                self._progress_tracker.finish_epoch()
                sys.stderr.write("After progress tracker finish_epoch()\n")
                self._progress_listener.after_finish_epoch()
                sys.stderr.write("After progress listener after_finish_epoch()\n")
                return

            if num_gpus() > 1:
                sys.stderr.write('before synchronize\n')
                torch.cuda.synchronize(torch.distributed.get_rank())
                sys.stderr.write('after synchronize\n')
                sys.stderr.write('before barrier 2\n')
                nccl_barrier_on_cpu()
                sys.stderr.write('after barrier 2\n')

        raise RuntimeError("train iterator should be infinite")

    def _set_shuffle_seed(self, iter):
        random.seed(iter)
        random.seed(random.randint(0, 10000000))


def _proprocess_texts_transformer_decoder(texts, tokens_lengths, dictionary: Dictionary, max_speech_len):
    batch_size, speech_len = texts.size()
    if speech_len != max_speech_len:
        x = torch.zeros(batch_size, max_speech_len - speech_len).long().to(texts.device)
        texts = torch.cat((texts, x), dim=1)
    x = torch.zeros(batch_size, 1).long().to(texts.device)
    texts = torch.cat((texts, x), dim=1)
    texts[get_padding_mask(tokens_lengths, 1 + max_speech_len)] = dictionary.eos_id()
    # texts.scatter_(1, tokens_lengths.view(-1, 1), dictionary.eos_id())
    return texts


def compute_loss(objective, result: ModelResult, batch: SpeechBatch, dictionary: Dictionary):
    speakers_num = len(batch.tokens)
    texts = batch.tokens.copy()
    max_speech_len = torch.stack(batch.tokens_lengths, dim=-1).max()
    if result.model_context["transformer_decoder"]:
        for s in range(speakers_num):
            texts[s] = _proprocess_texts_transformer_decoder(
                texts[s], batch.tokens_lengths[s], dictionary, max_speech_len)

    if isinstance(objective, CrossEntropyLoss):
#         LOG.info('Computing loss...')
        loss = 0
        for s in range(speakers_num):
#             LOG.info(f'CrossEntropyLoss {result.output[s]} vs {texts[s]}')
#             LOG.info(f'  shapes: {result.output[s].shape} vs {texts[s].shape}')
            obj = objective(logits=result.output[s], targets=texts[s])
            loss += obj
    elif isinstance(objective, CTCLossNM):
        loss = sum(objective(log_probs=result.log_probs[s], targets=texts[s], input_length=result.encoded_lengths,
                             target_length=batch.tokens_lengths[s]) for s in range(speakers_num))
    else:
        raise Exception(f"Unknown objective function: {objective}")

    return loss
