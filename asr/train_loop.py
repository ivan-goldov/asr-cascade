import sys

import json
import logging
import os
from typing import List

from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP

import torch
from torch.utils.tensorboard import SummaryWriter

import datetime

import torch.distributed
import torch.utils.data

from asr.models import EncoderDecoderModel
from asr.progress_tracker import ASRProgressTracker
from asr.train import ModelTrainer, MetricCalcer, TestMetricCalcer
from asr.disk_dataset import DiskTrainDataLayer, DiskTestDataset

from common.decoder import create_decoder
from common.dictionary import Dictionary
from common.objective import create_objective
from common.optimizer import create_optimizer
from common.text_processor import SentenceProcessor
from common.train_utils import AmpOptimizations, Optimization, ProgressListeners, ProgressSaver, restore_from_snapshot, \
    set_learning_rate, TorchOptimizerWrapper
from common.utils import num_gpus, gpu_id, set_epoch_seed, TrainParams

from lm.models.gpt import load_from_checkpoint as load_lm_from_checkpoint

import wandb

logging.basicConfig(filename="logs.txt", level=logging.DEBUG)
LOG = logging.getLogger()

def train_disk(local_rank: int,
               train_directories: List[str],
               test_table: str,
               model_definition: dict,
               dictionary: Dictionary,
               text_processor: str,
               features_config: dict,
               continue_model_path: str,
               language_model_path: str,
               snapshot_path: str,
               log_archive_path: str,
               params: TrainParams):
    
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = params.get_param("cudnn", False)
    
    LOG.debug(f"local_rank = {local_rank}")
    
    wandb.init(project="transformers", name=f"eval_loss_{local_rank}")
    
    # set up distributed training
    if local_rank is not None:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", rank=local_rank, world_size=int(os.environ["WORLD_SIZE"]))

    multi_gpu = torch.distributed.is_initialized()
    if multi_gpu:
        print(f"DISTRIBUTED TRAINING with {torch.distributed.get_world_size()} GPUs")
        LOG.debug(f"DISTRIBUTED TRAINING with {torch.distributed.get_world_size()} GPUs")

    # define amp optimiation level
    if params.get_param("fp16"):
        optimization_level = Optimization.mxprO1
    else:
        optimization_level = Optimization.mxprO0

    LOG.info("Model_config:")
    LOG.info(model_definition)

    LOG.info("Features config:")
    LOG.info(features_config)

    read_threads = params.get_param("read_threads")
    num_g = num_gpus()
    read_threads_per_gpu = (read_threads + num_g - 1) // num_g
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

    train_data_layer = DiskTrainDataLayer(directories=train_directories,
                                          batch_size=params.get_param("batch_size"),
                                          block_size=params.get_param("batch_block_size"),
                                          dictionary=dictionary,
                                          features_config=features_config,
                                          max_duration=params.get_param("max_duration"),
                                          wave_augmentation_config=params.get_param("wave_augmentations"),
                                          spec_augmentation_config=params.get_param("spec_augmentations"),
                                          latin_policy=params.get_param("latin_policy"),
                                          text_processor=text_processor,
                                          sort_by_length=params.get_param("sort_train_batch"),
                                          pad_to=params.get_param("pad_to"),
                                          read_threads=read_threads_per_gpu,
                                          seed=local_rank)
    
    if test_table:
        test_dataset = DiskTestDataset(directory=test_table,
                                       dictionary=dictionary,
                                       batch_size=params.get_param("batch_size"),
                                       features_config=features_config,
                                       sort_by_duration=params.get_param("sort_test_data"),
                                       in_memory=params.get_param("in_memory_test_data"),
                                       latin_policy=params.get_param("latin_policy"),
                                       text_processor=text_processor,
                                       read_threads=read_threads_per_gpu,
                                       pad_to=params.get_param("pad_to"))
    else:
        test_dataset = None

    if language_model_path:
        language_model, language_model_definition = load_lm_from_checkpoint(language_model_path)
        model_definition["language_model"] = language_model_definition
    else:
        language_model = None

    model = EncoderDecoderModel(model_definition, features_config["num-mel-bins"], dictionary, language_model).cuda()
    LOG.info(f"Number of parameters in encoder: {model.encoder.num_weights()}")
    LOG.info(f"Number of parameters in decode: {model.decoder.num_weights()}")
    LOG.info(f"Total number of parameters: {model.num_weights()}")

    objective = create_objective(dictionary, params)
    LOG.info("Objective created")
    optimizer = create_optimizer(model, params.get_param("optimizer"))
    LOG.info("Optimizer created")
    
    progress_tracker = ASRProgressTracker()
    LOG.info("Progress tracker created")
    progress_listener = ProgressListeners()
    LOG.info("Progress listener created")
    
    if snapshot_path is not None and os.path.exists(snapshot_path):
        LOG.info(f"Restoring from checkpoint {snapshot_path}")
        restore_from_snapshot(path=snapshot_path,
                              model=model,
                              optimizer=optimizer,
                              scheduler=None,
                              progress=progress_tracker)
        train_data_layer.set_progress(progress_tracker.epoch_reader_state())
        if not progress_tracker.is_same_data_sources(train_data_layer.data_sources()):
            raise RuntimeError("Different data source with sam snapshot, can't continue training")
    elif continue_model_path is not None:
        LOG.info(f"Will continue model {continue_model_path}")
        restore_from_snapshot(path=continue_model_path,
                              model=model,
                              optimizer=optimizer if params.get_param("continue_optimizer", False) else None,
                              scheduler=None,
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
        
#     if optimization_level in AmpOptimizations:
#         model, optimizer = amp.initialize(min_loss_scale=1.0, models=model, optimizers=optimizer,
#                                          opt_level=AmpOptimizations[optimization_level])
    model = DDP(model, device_ids=[local_rank]) if multi_gpu else model
    
    # Not here
    
    optimizer = TorchOptimizerWrapper(model.parameters(), optimizer, optimization_level, params.get_param("optimizer")["max_grad_norm"])

    decoder = create_decoder(nn_decoder=params.get_param("nn_decoder"), text_decoder=params.get_param("text_decoder"),
                             dictionary=dictionary)
    summary_writer = SummaryWriter(log_dir=progress_tracker.log_dir()) if not gpu_id() else None
    progress_listener.add(MetricCalcer(decoder=decoder, summary_writer=summary_writer))

    progress_saver = None
    if snapshot_path is not None:
        progress_saver = ProgressSaver(progress_tracker,
                                       model,
                                       optimizer,
                                       None,
                                       params.get_param("save_frequency"),
                                       snapshot_path,
                                       summary_writer.log_dir if summary_writer is not None else None,
                                       log_archive_path)
        if params.get_param("save_frequency") > 0:
            progress_listener.add(progress_saver)
    # Not here
      
    LOG.info("Starting .....")

    if test_dataset is not None:
        test_metric_calcer = TestMetricCalcer(test_dataset, model, objective, decoder, summary_writer)
        progress_listener.add(test_metric_calcer)

    if progress_tracker.is_training_finished():
        params_copy = dict(params.save())
        params_copy["features_config"] = features_config
        params_copy["model_definition"] = model_definition
        params_copy["dictionary"] = dictionary
        progress_tracker.on_new_training(params_copy)

    # Not here
        
    epochs = params.get_param("epochs")
    if params.get_param("epochs_policy") == "additional":
        epochs += progress_tracker.previous_training_epochs()

    # Not here
        
    trainer = ModelTrainer(model, objective, optimizer, progress_tracker, progress_listener)
    while True:
        
        # Not here
        
        seed = local_rank if local_rank is not None else 0 #params.get_param("seed")
        set_epoch_seed(seed, progress_tracker.epoch())

        LOG.debug("Yet another epoch\n")

        #Actually here
        
        # with amp.autocast():
        trainer.epoch(data_loader=train_data_layer.data_loader,
                      dictionary=dictionary,
                      epoch_duration_limit_seconds=params.get_param("epoch_duration_hours") * 60 * 60,
                      batch_log_frequency=params.get_param("train_frequency"),
                      test_log_frequency=params.get_param("test_frequency"))

        LOG.debug("Check output before THAT LINE\n")
        # LOG.debug(f"Epoch {progress_tracker.epoch()} finished at {datetime.now()}\n")

        if progress_tracker.epoch() >= epochs:
            break

    progress_tracker.finish_training()

    if snapshot_path is not None:
        progress_saver.on_training_finish()

    LOG.info("Training was finished")
