import time
import logging

import torch
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP

from common.disk_utils import BatchBlock
from common.objective import create_objective
from common.optimizer import create_optimizer
from common.train_utils import ProgressListeners, TorchOptimizerWrapper, Optimization
from common.train_utils import save_snapshot, restore_from_snapshot
from asr.disk_dataset import DiskTrainDataLayer
from asr.models import EncoderDecoderModel, ModelResult
from asr.progress_tracker import ASRProgressTracker
from asr.train import ModelTrainer, MetricCalcer, TestMetricCalcer, compute_loss

import wandb
import os

logging.basicConfig(filename="logs1.txt", level=logging.DEBUG)
LOG = logging.getLogger()


def train(local_rank, model_definition, features_config, dictionary, params):
    wandb.init(project="transformers", name=f"eval_loss_{local_rank}")

    if local_rank is not None:
        multi_gpu = True
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", rank=local_rank,
                                             world_size=int(os.environ["WORLD_SIZE"]))
        LOG.debug(f"local_rank = {local_rank}")
    else:
        multi_gpu = False

    train_data_layer = DiskTrainDataLayer(directories=['../train-clean-100'],  # directories,
                                          batch_size=params.get_param("batch_size"),
                                          block_size=params.get_param("batch_block_size"),
                                          dictionary=dictionary,
                                          features_config=features_config,
                                          max_duration=params.get_param("max_duration"),
                                          wave_augmentation_config=params.get_param("wave_augmentations"),
                                          spec_augmentation_config=params.get_param("spec_augmentations"),
                                          latin_policy=params.get_param("latin_policy"),
                                          text_processor=None,
                                          sort_by_length=params.get_param("sort_train_batch"),
                                          pad_to=params.get_param("pad_to"),
                                          read_threads=4,
                                          seed=local_rank)

    model = EncoderDecoderModel(model_definition, features_config["num-mel-bins"], dictionary, None).cuda()
    model = DDP(model, device_ids=[local_rank]) if multi_gpu else model

    objective = create_objective(dictionary, params)
    optimizer = create_optimizer(model, params.get_param("optimizer"))

    optimizer = TorchOptimizerWrapper(model.parameters(), optimizer, Optimization.mxprO1, 10.0)

    progress_tracker = ASRProgressTracker()
    progress_listener = ProgressListeners()

    restore_from_snapshot(path='../snapshot/snap_new.cpkt',
                          model=model,
                          optimizer=optimizer,
                          scheduler=None,
                          progress=progress_tracker)

    j = 0
    max_iter = 100000

    print("Started loading data")
    LOG.debug("Started loading data")

    start = time.time()

    for speech_batch_block in train_data_layer._data_loader:
        speech_batch_block: BatchBlock = speech_batch_block[0]
        for batch1, batch2 in speech_batch_block:
            if j == 0:
                print(f"loaded in {time.time() - start} seconds")
                LOG.debug(f"loaded in {time.time() - start} seconds")

            j += 1

            model.train()
            batch1 = batch1.cuda()
            batch2 = batch2.cuda()

            with amp.autocast():
                result: ModelResult = model(batch1, batch2)
                loss = compute_loss(objective, result, batch1, batch2, dictionary)
                loss /= len(batch1)

            wandb.log({'loss': loss})
            LOG.debug(f"loss={loss}")
            if (j + 1) % 100 == 0:
                print(f"{time.time():} loss={loss}")

            if (j + 1) % 10000 == 0:
                save_snapshot(model, optimizer, None,
                              progress_tracker,
                              snapshot_path="../snapshot/snap_new.cpkt", log_dir=None,
                              log_archive_path="tensorboard.tar")

            optimizer(loss)

            if j >= max_iter:
                break
        if j >= max_iter:
            break
