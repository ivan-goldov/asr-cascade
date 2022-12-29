import os
import torch
import logging
from asr.disk_dataset import DiskTrainDataLayer

logging.basicConfig(filename="test_logs.txt", level=logging.DEBUG)
LOG = logging.getLogger()


def test(local_rank, params_, dictionary, features_config):
    if local_rank is not None:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", rank=local_rank,
                                             world_size=int(os.environ["WORLD_SIZE"]))

    train_data_layer = DiskTrainDataLayer(
        directories=['../train-clean-100'],
        batch_size=params_.get_param("batch_size"),
        block_size=params_.get_param("batch_block_size"),
        dictionary=dictionary,
        features_config=features_config,
        max_duration=params_.get_param("max_duration"),
        wave_augmentation_config=params_.get_param("wave_augmentations"),
        spec_augmentation_config=params_.get_param("spec_augmentations"),
        latin_policy=params_.get_param("latin_policy"),
        text_processor=None,
        sort_by_length=params_.get_param("sort_train_batch"),
        pad_to=params_.get_param("pad_to"),
        read_threads=4,
        seed=local_rank)

    for speech_bath_block in train_data_layer._data_loader:
        for batch in speech_bath_block[0]:
            LOG.debug(f"id={local_rank} features={batch._features}")
            break
        break
