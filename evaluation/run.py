import json
import torch
import os
import pandas as pd
import sys
import numpy as np
from tqdm import tqdm

sys.path.append('/Users/rediska/PycharmProjects/transformer-the-greatest')

from asr.diarization_dataset import MultiTableDataset
from asr.validation_diarization_dataset import ExampleTestDataset, Kekos
from asr.models.encoder_decoder import EncoderDecoderModel
from asr.metrics import speaker_independent_word_error_rate
from common.dictionary import Dictionary, SentencePieceDict
from common.disk_utils import LatinPolicy
from common.train_utils import restore_from_snapshot, get_latest_snapshot_path
from common.text_processor import FilterBadTokensTextProcessor
from evaluation.evaluation_method.evaluation_method import EvaluationMethod
from evaluation.evaluation_method.teacher_forced_evaluation import TeacherForcedEvaluation
from evaluation.evaluation_method.greedy_evaluation import GreedyEvaluation
from evaluation.evaluation_method.beam_evaluation import BeamEvaluation
from evaluation.evaluation_dataset import EvaluationDataset, TableInfo
from asr.evaluation.generator import PowNorm
from typing import List, Optional, Tuple, Dict

N = 1  # 3
BATCH_SIZE = 24  # 24

MAX_SPEAKERS_NUM = 2

EVALUATION_METHODS: List[EvaluationMethod] = [
#     TeacherForcedEvaluation(),
    GreedyEvaluation(),
#     BeamEvaluation(beam_size=30, norm=PowNorm(3)),
]

EVALUATION_DATASETS: List[EvaluationDataset] = [
#     EvaluationDataset('Multichannel',
#                       TableInfo('kekos:data/validation_data/multichannel_test.json',
#                                 cut_range=(0.0, 1.0))),
#     EvaluationDataset('Lectures',
#                       TableInfo('example:data/validation_data/lectures/data.json',
#                                 cut_range=(0.0, 1.0))),
    EvaluationDataset('CommonVoice',
                      TableInfo('yt-raw:data/validation_data/common_voice_test.json',
                                cut_range=(0.0, 1.0))),
    EvaluationDataset('PhoneAcoustic',
                      TableInfo('yt-raw:data/yt_data/ru_phone_acoustic_shuffled.json',
                                cut_range=(0.9, 1.0))),
    EvaluationDataset('VoiceRecorderAcoustic',
                      TableInfo('yt-raw:data/yt_data/ru_voice_recorder_acoustic_shuffled.json',
                                cut_range=(0.9, 1.0))),
    EvaluationDataset('Silence',
                      TableInfo('yt-raw:data/yt_data/silence.json',
                                cut_range=(0.9, 1.0))),
    EvaluationDataset('SearchStroka',
                      TableInfo('yt-raw:data/yt_data/no_alice_search_stroka.json',
                                cut_range=(0.9, 1.0))),
    EvaluationDataset('Assistant',
                      TableInfo('yt-raw:data/yt_data/no_alice_assistant.json',
                                cut_range=(0.9, 1.0))),
    EvaluationDataset('Navi',
                      TableInfo('yt-raw:data/yt_data/no_alice_navi.json',
                                cut_range=(0.9, 1.0))),
    EvaluationDataset('Geq40',
                      TableInfo('yt-raw:data/yt_data/no_alice_texts_geq40.json',
                                cut_range=(0.9, 1.0))),
    EvaluationDataset('Spotter',
                      TableInfo('yt-raw:data/yt_data/no_alice_spotter.json',
                                cut_range=(0.9, 1.0))),
    EvaluationDataset('Quasar',
                      TableInfo('yt-raw:data/yt_data/no_alice_quasar.json',
                                cut_range=(0.9, 1.0))),
    EvaluationDataset('Rest',
                      TableInfo('yt-raw:data/yt_data/no_alice_rest.json',
                                cut_range=(0.9, 1.0))),
]

SNAPSHOTS = ['snapshot180000']


def construct_dataset(dataset: EvaluationDataset, dictionary: Dictionary, features_config: Dict, speakers_num: int, constant_gap: float):
    spec_augmentation_config = {
        "phone_aug": {
            "prob": 1.0,
            "alpha_from": 0.01,
            "alpha_to": 0.4,
            "height_from": 52,
            "height_to": 63,
            "width_from": 5,
            "width_to": 45,
            "fill_prob": 0.5
        }}
    wave_augmentation_config = {}
    test_dataset_args = {
        'dictionary': dictionary,
        'features_config': features_config,
        'batch_size': BATCH_SIZE,
        'block_size': 1,
        'max_duration': 16,
        'wave_augmentation_config': wave_augmentation_config,
        'spec_augmentation_config': spec_augmentation_config,
        'latin_policy': LatinPolicy.AsIs,
        'text_processor': FilterBadTokensTextProcessor(),
        'pad_to': 16,
        'sort_by_length': False,
        'merge_short_records': False,
        'max_speakers_num': MAX_SPEAKERS_NUM,
        'speakers_num_frequency': [(1 if s == speakers_num else 0) for s in range(1, MAX_SPEAKERS_NUM + 1)],
        'constant_gap': constant_gap,
    }
    if dataset.table.data_type in ['yt-raw', 'disk-raw']:
        return MultiTableDataset(
            tables=[(dataset.table.path, dataset.table.weight, dataset.table.cut_range)],
            **test_dataset_args)
    elif dataset.table.data_type in ['example']:
        return ExampleTestDataset(
            data_filepath=dataset.table.data_dir_path,
            **test_dataset_args)
    elif dataset.table.data_type in ['kekos']:
        return Kekos(
            json_table_filepath=dataset.table.data_dir_path,
            **test_dataset_args)
    else:
        raise IndexError(f'Data type {dataset.table.data_type} not supported')


def evaluate(model: EncoderDecoderModel, dictionary: Dictionary,
             dataset, evaluation_method: EvaluationMethod,
             transcriptions_logfile: Optional[str] = None) -> Tuple[float, List[List[str]], List[List[str]]]:
    all_hypotheses: List[List[str]] = []
    all_references: List[List[str]] = []
    for batchblock in tqdm(dataset.__iter__(max_sz=1), total=len(dataset)):
        for speechbatch in batchblock:
            if torch.cuda.is_available():
                speechbatch.cuda()
            hypotheses, tokenized_hypotheses, references = \
                evaluation_method.evaluate(model, dictionary, speechbatch)

            all_hypotheses += hypotheses
            all_references += references

    wer, _, _, data = speaker_independent_word_error_rate(
        all_hypotheses, all_references, return_data=True)

    wers1, wers2 = [], []
    if transcriptions_logfile:
        with open(transcriptions_logfile, 'w') as f:
            for i in range(0, len(data), 2):
                print(f'Sample {i}:', file=f)
                print(f'  Hyp1: {data[i][0]}', file=f)
                print(f'  Ref1: {data[i][1]}', file=f)
                print(f'  WER1: {data[i][2]}', file=f)
                print(f'  Hyp2: {data[i + 1][0]}', file=f)
                print(f'  Ref2: {data[i + 1][1]}', file=f)
                print(f'  WER2: {data[i + 1][2]}', file=f)
                wers1.append(data[i][2])
                wers2.append(data[i + 1][2])
                print(file=f)
            print(f'Mean WER 1st speaker: {sum(wers1) / len(wers1)}', file=f)
            print(f'Mean WER 2nd speaker: {sum(wers2) / len(wers2)}', file=f)
            print(f'Mean WER: {wer}', file=f)

    return wer, all_hypotheses, all_references


def transcriptions_filename(snapshot_filename: str, evaluation_method: EvaluationMethod,
                            dataset: EvaluationDataset, speakers_num: int, constant_gap: Optional[float], t: int) -> str:
    res = f'{snapshot_filename}_{evaluation_method.name()}_{dataset.name}'
    if dataset.is_synthetic():
        res += f'_{speakers_num}sp_{constant_gap}gap'
    res += f'_{t}.txt'
    return res


def run():
    # os.chdir('/home/rediska/transformer-the-greatest')
    results_dir = 'logs/evaluation'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    model_config = json.load(open('asr/configs/models/transformer10x6.json'))
    features_config = json.load(open('asr/configs/features/asr_default_64.json'))
    dictionary = SentencePieceDict('sp.model')

    outputs_df = pd.DataFrame([], columns=['WER', 'Snapshot', 'Evaluation method', 'Dataset', 'Speakers'])
    for snapshot in SNAPSHOTS:
        model = EncoderDecoderModel(
            model_config, features_config['num-mel-bins'], dictionary,
            max_speakers_num=MAX_SPEAKERS_NUM, language_model=None)
        model.eval()
        snapshot_filename = get_latest_snapshot_path('snapshots10x6') \
            if snapshot == 'latest' else os.path.join('snapshots10x6', snapshot)
        print(f'Snapshot: {snapshot_filename}')
        restore_from_snapshot('snapshots10x6', model, snapshot_path=snapshot_filename)
        snapshot_filename = snapshot_filename.split('/')[-1]
        if torch.cuda.is_available():
            model = model.cuda()

        for dataset in EVALUATION_DATASETS:
            for speakers_num in range(1 if dataset.is_synthetic() else MAX_SPEAKERS_NUM, MAX_SPEAKERS_NUM + 1):
                for constant_gap in ([None] if speakers_num == 1 or not dataset.is_synthetic() else [0.0, 0.5, 0.9]):
                    print(f' Dataset: {dataset.name} with {speakers_num} speakers and {constant_gap} gap')

                    for evaluation_method in EVALUATION_METHODS:
                        np.random.seed(47)
                        print(f'  Evaluation: {evaluation_method.name()}')
                        for t in range(N):
                            torch_dataset = construct_dataset(dataset, dictionary, features_config, speakers_num, constant_gap)
                            wer, hypotheses, references = \
                                evaluate(
                                    model, dictionary, torch_dataset, evaluation_method,
                                    transcriptions_logfile=os.path.join(
                                        results_dir,
                                        transcriptions_filename(snapshot_filename, evaluation_method,
                                                                dataset, speakers_num, constant_gap, t)))
                            print(f'   Total WER: {wer}')

                            outputs_df = outputs_df.append({
                                'WER': wer,
                                'Speakers': speakers_num,
                                'Evaluation method': evaluation_method.name(),
                                'Snapshot': snapshot_filename,
                                'Dataset': dataset.name,
                            }, ignore_index=True)
                            outputs_df.to_csv(os.path.join(results_dir, 'results.csv'))
