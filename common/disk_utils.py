from enum import Enum
import os
import sys
import time
from typing import List, Tuple, Generator
import itertools
from itertools import takewhile, repeat
import collections

import cyrtranslit
import numpy as np
import soundfile as sf

import torch

import logging

from common.utils import num_gpus, gpu_id

LOG = logging.getLogger()

# from ytreader import YTTableParallelReader


class LatinPolicy(str, Enum):
    AsIs = "as_is"
    CyrTranslit = "cyr_translit"
    Words = "words"
    Letters = "letters"
    Skip = "skip"


def load_text_from_yt_row(yt_row: dict, latin_policy: LatinPolicy) -> str:
    text = yt_row["text"]
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
                #sys.stderr.write(f"Reader #{reader_id} from {reader} will continue from {total}\n")
        return total


def clone_tensor_to_cuda(tensor):
    return tensor.clone() if (not torch.cuda.is_available() or tensor.is_cuda) else tensor.cuda()


class YtTableInfo:
    def __init__(self, path: str):
        self._path = path
        if "weight=" == path[0:len("weight=")]:
            weight_spec, path = path.split(":", maxsplit=1)
            _, w = weight_spec.split("=")
            self._weight = float(w)
        else:
            self._weight = 1

        self._data_type, table_path = path.split(":", 1)
        self._cluster, self._table = table_path.split("/", 1)

    def path(self):
        return self._path

    @property
    def table(self):
        return self._table

    @property
    def cluster(self):
        return self._cluster

    @property
    def data_type(self):
        return self._data_type

    @property
    def weight(self):
        return self._weight

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, YtTableInfo):
            return False
        return self._path == o._path


def load_text_from_record(record: dict, latin_policy: LatinPolicy, field_name="text") -> str:
    text = record[field_name]
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


# def genetate_random_sample(directories: List[str], seed: int = 0) -> dict:
#     rnd = np.random.RandomState(seed)
#     dir_name = rnd.choice(directories)
#     subdir = rnd.choice(list(os.walk(dir_name)))
#

def dirs_filenames(directories: List[str]):
    filenames = []
    references = []
    exts = ['.flac', '.wav', '.raw', '.ogg', '.mp3']
    for directory in directories:
        for p, d, f in os.walk(directory):
            for filename in f:
                if filename[-4:] == '.txt':
                    full_filename = os.path.join(p, filename)
                    with open(full_filename, 'r') as f:
                        for line in f.readlines():
                            words = line.split()
                            id = words[0]
                            text = ' '.join(words[1:])
                            for ext in exts:
                                speech_filename = os.path.join(p, id) + ext
                                if os.path.exists(speech_filename):
                                    break
                            if not os.path.exists(speech_filename):
                                LOG.warning(f'File {os.path.splitext(speech_filename)[0]} does not exist')
                                continue
                            references.append(text)
                            filenames.append(speech_filename)
    return filenames, references

def dir_files_generator(dir_name: str, seed: int = 0) -> Generator[dict, None, None]:
    LOG.debug(f"id={gpu_id()} seed={seed}")
    rnd = np.random.RandomState(seed)
    subdirs = list(os.walk(dir_name))
    rnd.shuffle(subdirs)
    for p, d, f in subdirs:
        filenames = f
        rnd.shuffle(filenames)
        for filename in filenames:
            if filename[-4:] == '.txt':
                full_filename = os.path.join(p, filename)
                with open(full_filename) as file:
                    if torch.distributed.is_initialized():
                        LOG.debug(f"id={gpu_id()}, open file {full_filename}")
                    else:
                        LOG.debug(f"id={0}, open file {full_filename}")
                    lines = file.readlines()
                    rnd.shuffle(lines)
                    for line in lines:
                        words = line.split()
                        id = words[0]
                        text = ' '.join(words[1:])
                        speech_filename = os.path.join(p, id) + os.path.extsep + 'flac'
                        with open(speech_filename, 'rb') as f:
                            data, samplerate = sf.read(f)
                        yield {"data": data, "sample_rate": samplerate, "text": text}, "disk-raw"


def identity(x):
    return x


def rawincount(filename):
    f = open(filename, 'rb')
    bufgen = takewhile(identity, (f.raw.read(1024 * 1024) for _ in repeat(None)))
    return sum(buf.count(b'\n') for buf in bufgen)

                        
def get_dir_samples_number(dir_name: str) -> int:
    #sys.stderr.write(f'{dir_name}\n')
    cnt = 0
    for p, d, f in os.walk(dir_name):
        for filename in f:
            if filename[-4:] == '.txt':
                full_filename = os.path.join(p, filename)
                cnt += rawincount(full_filename)
    return cnt


def skip(iterable, at_start=0, at_end=0):
    it = iter(iterable)
    for x in itertools.islice(it, at_start):
        pass
    queue = collections.deque(itertools.islice(it, at_end))
    for x in it:
        queue.append(x)
        yield queue.popleft()


class DiskDirReader:
    def __init__(self, directory: str, seed: int = 0):
        self._directory = directory
        self._cur = 0
        self._seed = seed
        self._iterator = iter(dir_files_generator(self._directory, self._seed))
        
    def reset_to_row(self, n_row: int):
        self._iterator = iter(skip(dir_files_generator(self._directory, self._seed), n_row, 0))
        self._cur = n_row
        
    def save(self) -> int:
        return self._cur
    
    def restore(self, progress: int):
        self.reset_to_row(progress)
        
    def make_subset_reader(self, offset_left, offset_right):
        reader = DiskDirReader(self._directory, self._seed)
        reader._iterator = iter(skip(dir_files_generator(self._directory, self._seed), offset_left, offset_right))
        return reader
        
    def __iter__(self):
        return self
    
    def __next__(self):
        self._cur += 1
        sample = next(self._iterator, None)
        if sample is None:
            raise StopIteration()
        return sample


class MultiDirsDiskReader:
    def __init__(self, directories: List[str], reader_id: int, total_readers: int, seed: int = None):
        self._directories = directories
        self._read_thread_id = reader_id
        self.filenames, self.references = dirs_filenames(directories)

        self._readers = []
        if seed is None:
            self._seed = gpu_id() * total_readers + reader_id
        else:
            self._seed = seed * total_readers + reader_id
        self.rnd = np.random.RandomState(self._seed)
        # for dir_name in self._directories:
        #     reader = DiskDirReader(dir_name, seed)
        #     reader.reset_to_row((get_dir_samples_number(dir_name) // total_readers) * reader_id)
        #     self._readers.append(reader)
        #self._reader_weights = np.array([source.weight for source in self._directories])
        #self._reader_weights /= np.sum(self._reader_weights)
        #self._reader_weights = np.ones(len(self._directories), dtype=np.float64) / len(self._directories)

    # def _pull_reader(self, reader_idx: int) -> dict:
    #     yt_row, _ = next(self._readers[reader_idx], (None, None))
    #     if yt_row is None:
    #         self._readers[reader_idx].reset_to_row(0)
    #         yt_row, _ = next(self._readers[reader_idx])
    #     return yt_row

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[dict, str]:
        # reader_idx = np.random.choice(len(self._reader_weights), p=self._reader_weights)
        # try:
        #     return self._pull_reader(reader_idx), "disk-raw"
        # except StopIteration as e:
        #     sys.stderr.write(f"Error reader_idx={reader_idx} dir={self._directories[reader_idx]}\n")
        id = self.rnd.choice(len(self.filenames))
        speech_filename, text = self.filenames[id], self.references[id]
        with open(speech_filename, 'rb') as f:
            data, samplerate = sf.read(f)
        return {"data": data, "sample_rate": samplerate, "text": text}, "disk-raw"

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

