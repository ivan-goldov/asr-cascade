from enum import Enum
import sys
import os
import time
import json
import math
from typing import List, Tuple, Dict
from pydub import AudioSegment

import cyrtranslit
import logging
import numpy as np
from common.utils import num_gpus, gpu_id

import torch
import boto3
import io

LOG = logging.getLogger()


class S3Loader:
    def __init__(self):
        self._aws_access_key = 'lA1lnViCN3jrt_xuBhIm'
        self._aws_secret_key = '7YSUeWPZmoRigsfLaNs_HmctVPaUCOSKXLo6-uZ7'
        self._s3 = None

    def _init_session(self):
        if self._s3 is not None:
            return
        session = boto3.session.Session(aws_access_key_id=self._aws_access_key,
                                        aws_secret_access_key=self._aws_secret_key)
        self._s3 = session.client(service_name='s3', endpoint_url='https://storage.yandexcloud.net')

    def load_from_yt_row(self, yt_row, output_sampling_rate):
#         LOG.info(f'Will load {yt_row}')
        if 'spec' in yt_row.keys():
            return self.load(yt_row['spec']['audio_encoding'], None,
                             yt_row['spec']['sample_rate_hertz'], yt_row['s3_obj']['bucket'],
                             yt_row['s3_obj']['key'], output_sampling_rate)
        else:
            self._init_session()
            s3_object = self._s3.get_object(Bucket=yt_row['s3_obj']['bucket'], Key=yt_row['s3_obj']['key'])
            data = AudioSegment.from_wav(io.BytesIO(s3_object['Body'].read()))
            return data.get_array_of_samples()

    def load(self, audio_encoding, audio_channel_count, sample_rate_hertz, bucket, key, output_sampling_rate):
#         LOG.info(f'Will load {(audio_encoding, audio_channel_count, sample_rate_hertz, bucket, key, output_sampling_rate)}')
        self._init_session()
        # assert audio_encoding in [1, 2, 3], 'expected PCM/OGG/WAV format'
        # assert audio_channel_count == 1, 'expected 1 channel'

        sampling_rate = sample_rate_hertz
        s3_object = self._s3.get_object(Bucket=bucket, Key=key)
        if audio_encoding == 1:
            data = AudioSegment.from_raw(s3_object['Body'], channels=1, sample_width=2,
                                         frame_rate=sampling_rate)
        elif audio_encoding == 3:
            data = AudioSegment.from_wav(io.BytesIO(s3_object['Body'].read()))
        else:
            data = AudioSegment.from_ogg(s3_object['Body'])
        if sampling_rate != output_sampling_rate:
            data = data.set_frame_rate(int(output_sampling_rate))
        return data.get_array_of_samples()


class LatinPolicy(str, Enum):
    AsIs = "as_is"
    CyrTranslit = "cyr_translit"
    Words = "words"
    Letters = "letters"
    Skip = "skip"


def load_text_from_yt_row(yt_row: dict, latin_policy: LatinPolicy) -> str:
    if isinstance(yt_row['ref'], str):
        text = yt_row['ref']
    else:
        assert len(yt_row['ref']) == 1
        text = yt_row['ref'][0]
    if latin_policy == LatinPolicy.CyrTranslit:
        return cyrtranslit.to_cyrillic(text, "ru")
    elif latin_policy == LatinPolicy.Skip:
        as_cyrillic = cyrtranslit.to_cyrillic(text, "ru")
        return "" if as_cyrillic != text else text
    elif latin_policy == LatinPolicy.Words:
        return " " + " ".join(["*"] * len(text.split(" "))) + " "
    elif latin_policy == LatinPolicy.Letters:
        to_align = [c if c.isspace() else "*" for c in text]
        return " " + "".join(to_align) + " "
    else:
        return text


class ReadThreadState:
    def __init__(self, read_thread_id: int = 0, state=None, timestamp: int = 0):
        self._read_thread_id = read_thread_id
        self._state = state
        self._timestamp = timestamp

    @property
    def read_thread_id(self) -> int:
        return self._read_thread_id

    @property
    def state(self):
        return self._state

    @property
    def timestamp(self) -> int:
        return self._timestamp

    def state_dict(self) -> dict:
        return {
            "read_thread_id": self._read_thread_id,
            "state": self._state,
            "timestamp": self._timestamp
        }

    def load_state_dict(self, state):
        self._read_thread_id = state["read_thread_id"]
        self._state = state["state"]
        self._timestamp = state.get("timestamp", 0)

    def __str__(self):
        stateStr = ";".join(self._state)
        return f"Reader #{self._read_thread_id}: {stateStr} (timestamp: {self._timestamp})"


class ReaderProgress:
    def __init__(self):
        self._states = {}

    def update(self, new_state: ReadThreadState, filter_old: bool = False):
        prev_state = self._states.get(new_state.read_thread_id, new_state)
        if filter_old and prev_state.timestamp > new_state.timestamp:
            return
        assert prev_state.timestamp <= new_state.timestamp
        self._states[new_state.read_thread_id] = new_state

    def states(self):
        return self._states

    def state(self, thread_id: int):
        return self._states[thread_id]

    def merge_progress(self, progress: list, filter_old: bool = False):
        for state in progress:
            reader_state = ReadThreadState()
            reader_state.load_state_dict(state)
            self.update(reader_state, filter_old)

    def save(self) -> List[dict]:
        return [state.state_dict() for state in self._states.values()]

    def filter_ranks(self, rank: int, total_ranks: int):
        keys_to_drop = [x for x in self._states.keys() if x % total_ranks != rank]
        for key in keys_to_drop:
            self._states.pop(key)

    def set_progress(self, progress: list):
        self._states = {}
        self.merge_progress(progress)
        return self

    def write_progress_to_readers(self, readers: list, reader_id: int):
        total = 0
        if reader_id in self._states.keys():
            assert len(readers) == len(self._states[reader_id].state)
            for reader, progress in zip(readers, self._states[reader_id].state):
                reader.set_progress(progress)
                sys.stderr.write(f"Reader #{reader_id} from {reader} will continue from {total}\n")
        return total


def clone_tensor_to_cuda(tensor):
    return tensor.clone() if tensor.is_cuda else tensor.cuda()


class YtReader:
    def __init__(self, table_path: str, start_position: int, end_position: int):
        assert start_position <= end_position
        LOG.info(f'create YtReader(start_position={start_position}, end_position={end_position})')
        self._state = {
            'table_path': table_path,
            'start_position': start_position,
            'end_position': end_position,
        }
        self._table_path = table_path
        self._table = None
        self._current_pos = None

    def _reset_current_pos(self):
        self._current_pos = np.random.randint(self._state['start_position'], self._state['end_position'])
        self._table.seek(self._current_pos)
        try:
            self._read_until_next_row()
        except:
            LOG.debug('Resetting reader position...\n')
            self._reset_current_pos()

    def reset_reader(self):
        self._table = open(self._table_path, 'r')
        self._reset_current_pos()

    def save(self):
        return self._state.copy()

    def restore(self, state: Dict):
        for key, value in self._state.items():
            assert key in state
        self._state = state
        self.reset_reader()

    def _read_until_next_row(self):
        s = self._table.read(1)
        if len(s) == 0:
            raise StopIteration
        self._current_pos += 1
        while s[-1] != '\n':
            s += self._table.read(1)
            self._current_pos += 1
            if self._current_pos >= self._state['end_position']:
                raise StopIteration
        return s

    def __next__(self):
        if self._table is None or self._current_pos is None:
            self.reset_reader()
        try:
            self._reset_current_pos()
            s = self._read_until_next_row()
            j = json.loads(s)
            if j['duration'] >= 16:  # TODO: блять)
                return self.__next__()
            return j
        except StopIteration:
            return self.__next__()

    def __iter__(self):
        return self


class DiskReader:
    def __init__(self, data, start_position: int, end_position: int):
        LOG.info(f'create DiskReader(start_position={start_position}, end_position={end_position})')
        assert start_position <= end_position
        self._state = {
            'start_position': start_position,
            'end_position': end_position,
        }
        self.data = data

    def save(self):
        return self._state.copy()

    def restore(self, state: Dict):
        for key, value in self._state.items():
            assert key in state
        self._state = state

    def __next__(self):
        idx = np.random.randint(self._state['start_position'], self._state['end_position'])
        # print(f'Read new at position {idx}')
        x = self.data[idx]
        return x

    def __iter__(self):
        return self


def create_datasource(data_desc: str, weight: float, start: float, end: float):
    data_type = data_desc.split(':')[0]
    data_path = data_desc[len(data_type) + 1:]
    if data_type == 'yt-raw':
        return YtDataSource(data_path, weight, start, end)
    elif data_type == 'disk-raw':
        return DiskDataSource(data_path, weight, start, end)
    elif data_type == 'kekos':
        return KekosDataSource(data_path, weight, start, end)
    else:
        raise Exception('блять)')


class DiskDataSource:
    def __init__(self, data_path: str, weight: float, start: float, end: float):
        with open(data_path) as f:
            data = json.loads(f.read())
        w = 'first' if 'first' in data_path else 'last' if 'last' in data_path else None
        for d in data:
            record_id = d['record_id']
            d['link'] = \
                f'https://storage.yandexcloud.net/voicerecorder/VoiceRecorder/{w}/{record_id}.wav'
        self._data = data
        self._weight = weight
        sz = len(self._data)
        self._start = math.ceil(sz * start)
        self._end = int(sz * end)

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def data_type(self) -> str:
        return 'disk-raw'

    @property
    def size(self) -> int:
        return self._end - self._start

    def create_reader(self, start_position: int = None, end_position: int = None) -> DiskReader:
        return DiskReader(data=self._data,
                          start_position=(start_position or 0) + self._start,
                          end_position=(end_position or self.size) + self._start)


class KekosDataSource:
    def __init__(self, data_path: str, weight: float, start: float, end: float):
        lines = open(data_path, 'r').readlines()
        self._data = [json.loads(l.rstrip()) for l in lines]
        self._weight = weight
        sz = len(self._data)
        self._start = math.ceil(sz * start)
        self._end = int(sz * end)

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def data_type(self) -> str:
        return 'kekos'

    @property
    def size(self) -> int:
        return self._end - self._start

    def create_reader(self, start_position: int = None, end_position: int = None) -> DiskReader:
        return DiskReader(data=self._data,
                          start_position=(start_position or 0) + self._start,
                          end_position=(end_position or self.size) + self._start)


class YtDataSource:
    def __init__(self, table_path: str, weight: float, start: float, end: float):
        self._table_path = table_path
        self._weight = weight

        sz = os.stat(table_path).st_size
        self._start = math.ceil(sz * start)
        self._end = int(sz * end)

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def data_type(self) -> str:
        return 'yt-raw'

    @property
    def size(self) -> int:
        return self._end - self._start

    def create_reader(self, start_position: int = None, end_position: int = None) -> YtReader:
        return YtReader(table_path=self._table_path,
                        start_position=(start_position or 0) + self._start,
                        end_position=(end_position or self.size) + self._start)


class MultiTableYtReader:
    def __init__(self, data_sources: List, reader_id: int, total_readers: int, rows_num: int):
        self._data_sources = data_sources
        self._read_thread_id = reader_id
        self._total_readers = total_readers
        self._readers = []
        LOG.info(f'Creating readers for thread {self._read_thread_id}/{self._total_readers}')
        for source in self._data_sources:
            reader = source.create_reader(
                start_position=source.size * self._read_thread_id // self._total_readers,
                end_position=source.size * (self._read_thread_id + 1) // self._total_readers)
            self._readers.append(reader)
        LOG.info(f'Readers created')
        self._reader_weights = np.array([source.weight for source in self._data_sources]).astype(float)
        self._reader_weights /= np.sum(self._reader_weights).astype(float)
        self._rows_num = rows_num

    def _pull_reader(self, reader_idx: int) -> List[dict]:
        rows = [next(self._readers[reader_idx], None) for _ in range(self._rows_num)]
        if any(row is None for row in rows):
            sys.stderr.write('Row is None, resetting readers...\n')
            self._readers[reader_idx] = self._data_sources[reader_idx].create_reader(
                start_position=self._data_sources[reader_idx].size * self._read_thread_id // self._total_readers,
                end_position=self._data_sources[reader_idx].size * (self._read_thread_id + 1) // self._total_readers)
            rows = [next(self._readers[reader_idx], None) for _ in range(self._rows_num)]
        return rows

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[List[dict], str]:
        reader_idx = np.random.choice(len(self._reader_weights), p=self._reader_weights)
        return self._pull_reader(reader_idx), self._data_sources[reader_idx].data_type

    def set_progress(self, thread_state: ReadThreadState):
        assert len(thread_state.state) == len(self._readers)
        for reader, progress in zip(self._readers, thread_state.state):
            reader.restore(progress)

    def get_progress(self) -> ReadThreadState:
        return ReadThreadState(self._read_thread_id,
                               [reader.save() for reader in self._readers],
                               int(time.time()))


class BatchBlock:
    def __init__(self, batches: list, reader_state: ReadThreadState):
        self._batches = batches
        self._reader_state = reader_state

    def cuda(self):
        self._batches = [batch.cuda() for batch in self._batches]
        return self

    def pin_memory(self):
        self._batches = [batch.pin_memory() for batch in self._batches]
        return self

    def __iter__(self):
        return iter(self._batches)

    @property
    def batches(self):
        return self._batches

    @property
    def read_thread_state(self):
        return self._reader_state


def readers_info():
    num_workers = 1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size()
    my_rank = 0 if not torch.distributed.is_initialized() else torch.distributed.get_rank()
    data_load_process_info = torch.utils.data.get_worker_info()
    if data_load_process_info is not None:
        load_threads = data_load_process_info.num_workers
        thread_id = data_load_process_info.id
    else:
        load_threads = 1
        thread_id = 0
    part_id = num_workers * thread_id + my_rank
    return int(part_id), num_workers * load_threads


def find_max_len(items, idx):
    return max(len(item[idx]) for item in items)
