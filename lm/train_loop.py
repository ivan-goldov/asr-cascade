import json
import logging
import os
from typing import List

from apex import amp
from apex.parallel import DistributedDataParallel as DDP

import torch
from torch.utils.tensorboard import SummaryWriter

from common.decoder import create_decoder
from common.dictionary import Dictionary
from common.lr_scheduler import create_scheduler
from common.objective import create_objective
from common.optimizer import create_optimizer
from common.progress_tracker import ProgressTracker
from common.text_processor import SentenceProcessor
from common.train_utils import AmpOptimizations, Optimization, ProgressListeners, ProgressSaver, restore_from_snapshot, \
    set_learning_rate, TorchOptimizerWrapper
from common.utils import num_gpus, gpu_id, set_epoch_seed, TrainParams

from lm.models import GPT
from lm.train import ModelTrainer, MetricCalcer, TestMetricCalcer
from lm.yt_dataset import YtTrainDataLayer, YtTestDataset

LOG = logging.getLogger()


def train_yt(local_rank: int,
             train_tables: List[str],
             test_table: str,
             model_definition: dict,
             dictionary: Dictionary,
             text_processor: str,
             continue_model_path: str,
             snapshot_path: str,
             log_archive_path: str,
             params: TrainParams):
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = params.get_param("cudnn")

    # set up distributed training
    if local_rank is not None:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    multi_gpu = torch.distributed.is_initialized()
    if multi_gpu:
        LOG.debug(f"DISTRIBUTED TRAINING with {torch.distributed.get_world_size()} GPUs")

    # define amp optimiation level
    if params.get_param("fp16"):
        optimization_level = Optimization.mxprO1
    else:
        optimization_level = Optimization.mxprO0

    LOG.info("Model_config:")
    LOG.info(model_definition)

    read_threads = params.get_param("read_threads")
    read_threads_per_gpu = (read_threads + num_gpus() - 1) // num_gpus()
    if read_threads_per_gpu * num_gpus() > read_threads:
        LOG.warning("Read threads per GPU is always rounded up, will use more threads than specified: "
                    f"read threads = {read_threads}; num gpus = {num_gpus()},"
                    f"will read in {read_threads_per_gpu * num_gpus()} threads")

    if text_processor:
        with open(text_processor) as f:
            text_processor_config = json.load(f)
        text_processor = SentenceProcessor(**text_processor_config)
    else:
        text_processor = None

    train_data_layer = YtTrainDataLayer(tables=train_tables,
                                        batch_size=params.get_param("batch_size"),
                                        block_size=params.get_param("batch_block_size"),
                                        dictionary=dictionary,
                                        max_num_words=params.get_param("max_num_words"),
                                        words_shift=params.get_param("words_shift"),
                                        max_num_tokens=params.get_param("max_num_tokens"),
                                        empty_samples_proportion=params.get_param("empty_samples_proportion"),
                                        latin_policy=params.get_param("latin_policy"),
                                        text_processor=text_processor,
                                        use_bos=params.get_param("use_bos"),
                                        use_eos=params.get_param("use_eos"),
                                        sort_by_length=params.get_param("sort_train_batch"),
                                        pad_to=params.get_param("pad_to"),
                                        read_threads=read_threads_per_gpu)
    if test_table:
        test_dataset = YtTestDataset(path=test_table,
                                     dictionary=dictionary,
                                     batch_size=params.get_param("batch_size"),
                                     latin_policy=params.get_param("latin_policy"),
                                     text_processor=text_processor,
                                     use_bos=params.get_param("use_bos"),
                                     use_eos=params.get_param("use_eos"),
                                     sort_by_duration=params.get_param("sort_test_data"),
                                     in_memory=params.get_param("in_memory_test_data"),
                                     read_threads=read_threads_per_gpu,
                                     pad_to=params.get_param("pad_to"))
    else:
        test_dataset = None

    model = GPT(model_definition, dictionary).cuda()
    LOG.info(f"Total number of parameters: {model.num_weights()}")

    objective = create_objective(dictionary, params)
    optimizer = create_optimizer(model, params.get_param("optimizer"))
    scheduler = create_scheduler(optimizer, params)

    progress_tracker = ProgressTracker()
    progress_listener = ProgressListeners()

    if snapshot_path is not None and os.path.exists(snapshot_path):
        LOG.info(f"Restoring from checkpoint {snapshot_path}")
        restore_from_snapshot(path=snapshot_path, model=model,
                              optimizer=optimizer,
                              scheduler=scheduler,
                              progress=progress_tracker)
        train_data_layer.set_progress(progress_tracker.epoch_reader_state())
        if not progress_tracker.is_same_data_sources(train_data_layer.data_sources()):
            raise RuntimeError("Different data source with sam snapshot, can't continue training")
    elif continue_model_path is not None:
        LOG.info(f"Will continue model {continue_model_path}")
        restore_from_snapshot(path=continue_model_path, model=model,
                              optimizer=optimizer if params.get_param("continue_optimizer", False) else None,
                              scheduler=scheduler if params.get_param("continue_optimizer", False) else None,
                              progress=progress_tracker)
        set_learning_rate(optimizer, params.get_param("optimizer")["lr"])
        if progress_tracker.is_same_data_sources(train_data_layer.data_sources()):
            train_data_layer.set_progress(progress_tracker.epoch_reader_state())
        if not progress_tracker.is_current_epoch_finished():
            LOG.warning("Previous epoch was not finished, will force finish now")
            progress_tracker.finish_epoch()
            progress_tracker.finish_training()

    if not progress_tracker.is_same_data_sources(train_data_layer.data_sources()):
        progress_tracker.on_new_data_sources(train_data_layer.data_sources())

    if optimization_level in AmpOptimizations:
        model, optimizer = amp.initialize(min_loss_scale=1.0, models=model, optimizers=optimizer,
                                          opt_level=AmpOptimizations[optimization_level])
    model = DDP(model) if multi_gpu else model
    optimizer = TorchOptimizerWrapper(optimizer, optimization_level, params.get_param("optimizer")["max_grad_norm"])

    decoder = create_decoder(nn_decoder=params.get_param("nn_decoder"), text_decoder=params.get_param("text_decoder"),
                             dictionary=dictionary)
    summary_writer = SummaryWriter(log_dir=progress_tracker.log_dir()) if not gpu_id() else None
    progress_listener.add(MetricCalcer(decoder=decoder, summary_writer=summary_writer))

    progress_saver = None
    if snapshot_path is not None:
        progress_saver = ProgressSaver(progress_tracker,
                                       model,
                                       optimizer,
                                       scheduler,
                                       params.get_param("save_frequency"),
                                       snapshot_path,
                                       summary_writer.log_dir if summary_writer is not None else None,
                                       log_archive_path)
        if params.get_param("save_frequency") > 0:
            progress_listener.add(progress_saver)

    if test_dataset is not None:
        test_metric_calcer = TestMetricCalcer(test_dataset, model, objective, decoder, summary_writer)
        progress_listener.add(test_metric_calcer)

    if progress_tracker.is_training_finished():
        params_copy = dict(params.save())
        params_copy["model_definition"] = model_definition
        params_copy["dictionary"] = dictionary
        progress_tracker.on_new_training(params_copy)

    num_iterations = params.get_param("num_iterations")
    if params.get_param("iterations_policy") == "additional":
        num_iterations += progress_tracker.previous_training_iterations()

    trainer = ModelTrainer(model, objective, optimizer, scheduler, progress_tracker, progress_listener)
    while progress_tracker.total_iterations() < num_iterations:
        seed = params.get_param("seed")
        set_epoch_seed(seed, progress_tracker.epoch())

        trainer.epoch(data_loader=train_data_layer.data_loader,
                      dictionary=dictionary,
                      num_iterations=num_iterations,
                      batch_log_frequency=params.get_param("train_frequency"),
                      test_log_frequency=params.get_param("test_frequency"))

        LOG.debug(f"Epoch {progress_tracker.epoch()} finished")

    progress_tracker.finish_training()

    if snapshot_path is not None:
        progress_saver.on_training_finish()

    LOG.info("Training was finished")
