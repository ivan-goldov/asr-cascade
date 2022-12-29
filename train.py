# Один мальчик родился с гайкой вместо пупка.

# И он спросил у родителей, почему у него там гайка. Родители пообещали ему рассказать об этом на его 14-летие.

# Ему исполнилось 14. И он опять подошёл и спросил у родителей, почему у него вместо пупка гайка.
# Родители пообещали рассказать ему об этом когда ему будет 18 лет.

# В 18 лет он спросил снова и родители рассказали ему, что есть один остров на котором растет пальма, а под этой пальмой зарыт сундук.

# Парень долго копил денег и всё таки приехал на этот остров. Нашёл пальму, откопал сундук, в котором лежала отвёртка.
# Он открутил гайку отвёрткой и у него отвалилась ЖОПА.

import logging
import json

from torch.nn.parallel import DistributedDataParallel as DDP

import torch.distributed
import torch.utils.data

from asr.models import EncoderDecoderModel
from asr.progress_tracker import ASRProgressTracker
from asr.train import ModelTrainer, MetricCalcer, TestMetricCalcer
from asr.diarization_dataset import TrainDataLayer, MultiTableDataset
from asr.validation_diarization_dataset import ExampleTestDataset, Kekos

from common.decoder import create_decoder
from common.objective.cross_entropy import CrossEntropyLoss
from common.optimizer import create_optimizer
from common.train_utils import Optimization, ProgressListeners, TorchOptimizerWrapper, ProgressSaver
from common.utils import set_epoch_seed
from common.dictionary import SentencePieceDict
from common.disk_utils import LatinPolicy
from common.text_processor import FilterBadTokensTextProcessor

from evaluation.evaluation_method.evaluation_method import EvaluationMethod

from typing import List, Optional, Dict, Tuple

import wandb
import os
import sys
import numpy as np

from common.train_utils import restore_from_snapshot

TEST_SIZE = 4  # 4
TEST_BLOCK_SIZE = 4  # 4
BATCH_SIZE = 24  # 24
TRAIN_BLOCK_SIZE = 16  # 16


class LoggerWriter:
    def __init__(self, level, filename):
        self.level = level
        with open(filename, 'w') as f:
            self._fileno = f.fileno()
        self.encoding = 'acsii'

    def write(self, message):
        if message != '\n':
            self.level(message)

    def flush(self):
        pass

    def fileno(self):
        return self._fileno


class SyntheticDataGenParams:
    def __init__(self, speakers_num_freq: List[float], relative_overlap: Optional[float] = None,
                 case_name: Optional[str] = None):
        self.speakers_num = len(speakers_num_freq)
        self.speakers_num_freq = speakers_num_freq
        self.relative_overlap = relative_overlap
        self.case_name = case_name


def get_cut_range(global_cut_range, local_rank, world_size):
    a, b = global_cut_range
    return (local_rank * (b - a) / world_size + a,
            (local_rank + 1) * (b - a) / world_size + a)


def train(local_rank,
          max_speakers_num,

          train_tables: List[Tuple[str, int]],
          synthetic_train_params: SyntheticDataGenParams,

          synthetic_test_tables: Dict[str, Tuple[List[Tuple[str, int]], List[EvaluationMethod], List[SyntheticDataGenParams]]],
          test_tables: Dict[str, Tuple[str, List[EvaluationMethod]]],

          train_frequency: int, test_frequency: int, save_frequency: int,
          snapshots_dir: str, snapshot_path: str,
          model_def_path: str, features_config_path: str,
          lr: float, epoch_duration_hours: int, restore_latest_snapshot: bool, epochs_num: int):
    log_filename = os.path.join('logs',
                                f'train_cuda:{local_rank}.txt' if local_rank is not None else 'train.txt')
    logging.basicConfig(filename=log_filename, level=logging.INFO)
    LOG = logging.getLogger()
    sys.stdout = LoggerWriter(LOG.info, log_filename)
    sys.stderr = LoggerWriter(LOG.info, log_filename)

    model_definition = json.load(open(model_def_path))
    features_config = json.load(open(features_config_path))
    dictionary = SentencePieceDict('sp.model')
    optimization_level = Optimization.mxprO0
    language_model = None
    text_processor = FilterBadTokensTextProcessor()

    torch.backends.cudnn.benchmark = False

    wandb.init(project="transformer10x6", name=f"eval_loss_{local_rank}")

    LOG.info(f"local_rank = {local_rank}")

    if local_rank is not None:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", rank=local_rank,
                                             world_size=int(os.environ["WORLD_SIZE"]))

    multi_gpu = torch.distributed.is_initialized()
    if multi_gpu:
        LOG.info(f"DISTRIBUTED TRAINING with {torch.distributed.get_world_size()} GPUs")

    train_cut_range = (0, 0.9)
    test_cut_range = (0.9, 1.0)
    if local_rank is not None:
        train_cut_range = get_cut_range((0, 0.9), local_rank, torch.distributed.get_world_size())
        test_cut_range = get_cut_range((0.9, 1.0), local_rank, torch.distributed.get_world_size())

    tables: List[Tuple[str, int, Tuple[float, float]]] = \
        [(t, w, train_cut_range) for t, w in train_tables]
    train_data_layer = TrainDataLayer(
        tables=tables,
        batch_size=BATCH_SIZE,
        block_size=TRAIN_BLOCK_SIZE,
        dictionary=dictionary,
        features_config=json.loads(open("asr/configs/features/asr_default_64.json").read()),
        max_duration=16,
        wave_augmentation_config={},
        spec_augmentation_config={"phone_aug": {
            "prob": 1.0,
            "alpha_from": 0.01,
            "alpha_to": 0.4,
            "height_from": 52,
            "height_to": 63,
            "width_from": 5,
            "width_to": 45,
            "fill_prob": 0.5
        }},
        latin_policy=LatinPolicy.AsIs,
        text_processor=text_processor,
        sort_by_length=False,
        merge_short_records=False,
        pad_to=16,
        read_threads=8,
        max_speakers_num=synthetic_train_params.speakers_num,
        speakers_num_frequency=synthetic_train_params.speakers_num_freq,
        overlap=synthetic_train_params.relative_overlap,
    )

    test_dataset_args = {
        'dictionary': dictionary,
        'features_config': json.loads(open("asr/configs/features/asr_default_64.json").read()),
        'batch_size': BATCH_SIZE,
        'block_size': TEST_BLOCK_SIZE,
        'max_duration': 16,
        'wave_augmentation_config': {},
        'spec_augmentation_config': {"phone_aug": {
            "prob": 1.0,
            "alpha_from": 0.01,
            "alpha_to": 0.4,
            "height_from": 52,
            "height_to": 63,
            "width_from": 5,
            "width_to": 45,
            "fill_prob": 0.5
        }},
        'latin_policy': LatinPolicy.AsIs,
        'text_processor': text_processor,
        'pad_to': 16,
        'sort_by_length': False,
        'merge_short_records': False,
    }

    test_datasets = {}
    # Add validation datasets
    for validation_table_name, (validation_table, evaluation_methods) in test_tables.items():
        print(f'Start reading {validation_table_name}...')
        if validation_table_name.startswith('lecture'):
            test_datasets[validation_table_name] = (ExampleTestDataset(
                **test_dataset_args,
                data_filepath=validation_table,
            ), evaluation_methods)
        else:
            test_datasets[validation_table_name] = (Kekos(
                **test_dataset_args,
                json_table_filepath=validation_table,
            ), evaluation_methods)
    # Add synthetic datasets
    for test_tables_block_name, (test_tables_block, evaluation_methods, generation_params) in synthetic_test_tables.items():
        for p in generation_params:
            print(f'Start reading {test_tables_block_name} with {p.case_name}...')
            name = test_tables_block_name + '_' + p.case_name
            assert name not in test_datasets.keys()
            test_datasets[name] = (MultiTableDataset(
                **test_dataset_args,
                speakers_num_frequency=p.speakers_num_freq,
                tables=[(table, weight, test_cut_range) for table, weight in test_tables_block],
                constant_gap=p.relative_overlap,
                max_speakers_num=p.speakers_num,
            ), evaluation_methods)

    model = EncoderDecoderModel(model_definition, features_config["num-mel-bins"], dictionary,
                                max_speakers_num=max_speakers_num, language_model=language_model)
    if torch.cuda.is_available():
        model = model.cuda()
    LOG.info(f"Number of parameters in encoder: {model.encoder.num_weights()}")
    LOG.info(f"Number of parameters in decoder: {model.decoder.num_weights()}")
    LOG.info(f"Total number of parameters: {model.num_weights()}")

    objective = CrossEntropyLoss(pad_id=dictionary.pad_id())
    optimizer = create_optimizer(model,
                                 {'optimizer_kind': 'adam', 'lr': lr, 'weight_decay': 0.01, 'max_grad_norm': 1.0})

    progress_tracker = ASRProgressTracker()
    LOG.info("Progress tracker created")
    progress_listener = ProgressListeners()
    LOG.info("Progress listener created")

    optimizer = TorchOptimizerWrapper(model.parameters(), optimizer, optimization_level, 1.0)

    decoder = create_decoder(nn_decoder="greedy", text_decoder="simple",
                             dictionary=dictionary)

    progress_saver = ProgressSaver(progress_tracker, model, optimizer, None, save_frequency, snapshots_dir,
                                   None, "tensorboard.tar")
    train_metric_calcer = MetricCalcer(decoder=decoder)
    test_metric_calcer = TestMetricCalcer(test_datasets, model=model, objective=objective, decoder=decoder,
                                          dictionary=dictionary, test_sz=TEST_SIZE)
    progress_listener.add(progress_saver)
    progress_listener.add(train_metric_calcer)
    progress_listener.add(test_metric_calcer)

    if progress_tracker.is_training_finished():
        params_copy = dict({})
        params_copy["features_config"] = features_config
        params_copy["model_definition"] = model_definition
        params_copy["dictionary"] = dictionary
        progress_tracker.on_new_training(params_copy)

    LOG.info("Starting .....")

    if restore_latest_snapshot:
        LOG.info("Will restore lastest snapshot")
        progress_tracker.finish_epoch()
        LOG.info("Restoring snapshot .....")
        restore_from_snapshot(snapshots_dir, model, optimizer, None, progress_tracker,
                              progress_saver, snapshot_path=snapshot_path)

    model = DDP(model, device_ids=[local_rank]) if multi_gpu else model

    trainer = ModelTrainer(model, objective, optimizer, progress_tracker, progress_listener)

    for ep in range(epochs_num):
        seed = local_rank if local_rank is not None else 0
        set_epoch_seed(seed, progress_tracker.epoch())

        LOG.info("Yet another epoch {}\n".format(progress_tracker.epoch()))

        trainer.epoch(data_loader=train_data_layer.data_loader,
                      dictionary=dictionary,
                      epoch_duration_limit_seconds=epoch_duration_hours * 60 * 60,
                      batch_log_frequency=train_frequency,
                      test_log_frequency=test_frequency)

        LOG.debug("Check output before THAT LINE\n")
