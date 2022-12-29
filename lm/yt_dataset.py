import copy
from functools import partial
from itertools import repeat, zip_longest
import logging
import math
import random
from typing import Optional

import torch.distributed as dist
from torch.utils.data import Dataset, IterableDataset

from common.dictionary import Dictionary
from common.utils import Timer
from common.text_processor import SentenceProcessor
from common.yt_utils import *

LOG = logging.getLogger()


def find_max_len(items, ind):
    max_len = -1
    for item in items:
        if item[ind].item() > max_len:
            max_len = item[ind].item()
    return max_len


class TextBatch:
    @staticmethod
    def create_from_list(batch: List, pad_to: int):
        batch_size = len(batch)

        max_tokens_length = max(x[2].item() for x in batch)
        max_tokens_length = ((max_tokens_length + pad_to - 1) // pad_to) * pad_to
        batched_source_tokens = torch.zeros(batch_size, max_tokens_length)
        batched_target_tokens = torch.zeros(batch_size, max_tokens_length)
        tokens_lengths = []

        for i, sample in enumerate(batch):
            batched_source_tokens[i].narrow(0, 0, sample[0].size(0)).copy_(sample[0])
            batched_target_tokens[i].narrow(0, 0, sample[1].size(0)).copy_(sample[1])
            tokens_lengths.append(sample[2])

        return TextBatch(batched_source_tokens, batched_target_tokens, torch.stack(tokens_lengths))

    def __init__(self, source_tokens, target_tokens, tokens_lengths):
        self._source_tokens = source_tokens.long()
        self._target_tokens = target_tokens.long()
        self._tokens_lengths = tokens_lengths.long()
        self._batch_size = self._source_tokens.size(0)

    def cuda(self):
        self._source_tokens = self._source_tokens.cuda()
        self._target_tokens = self._target_tokens.cuda()
        self._tokens_lengths = self._tokens_lengths.cuda()
        return self

    def pin_memory(self):
        self._source_tokens = self._source_tokens.pin_memory()
        self._target_tokens = self._target_tokens.cuda()
        self._tokens_lengths = self._tokens_lengths.pin_memory()
        return self

    def __len__(self):
        return self._batch_size

    @property
    def source_tokens(self):
        return self._source_tokens

    @property
    def target_tokens(self):
        return self._target_tokens

    @property
    def tokens_lengths(self):
        return self._tokens_lengths

    def clone_to_gpu(self):
        return TextBatch(clone_tensor_to_cuda(self._source_tokens),
                         clone_tensor_to_cuda(self._target_tokens),
                         clone_tensor_to_cuda(self._tokens_lengths))


class SampleParser:
    def __init__(self, dictionary: Dictionary, latin_policy: LatinPolicy, text_processor: Optional[SentenceProcessor],
                 use_bos: bool, use_eos: bool):
        self._dictionary = dictionary
        self._latin_policy = latin_policy
        self._text_processor = text_processor
        self._use_bos = use_bos
        self._use_eos = use_eos

    def get_text(self, yt_row: dict):
        text = load_text_from_yt_row(yt_row, self._latin_policy)
        if self._text_processor is not None:
            text = self._text_processor.process_sentence(text)
        return text

    def tokenize_text(self, text):
        bos_id = self._dictionary.bos_id()
        sil_id = self._dictionary.sil_id()
        eos_id = self._dictionary.eos_id()

        if not text:
            tokens = [bos_id, sil_id, eos_id]
        else:
            tokens = [bos_id] if self._use_bos else []
            tokens.extend(self._dictionary.encode(text))
            if self._use_eos:
                tokens.append(eos_id)

        source_tokens = torch.tensor(tokens[:-1])
        target_tokens = torch.tensor(tokens[1:])
        tokens_len = torch.tensor(len(source_tokens)).int()

        return source_tokens, target_tokens, tokens_len

    def parse(self, yt_row):
        text = self.get_text(yt_row)
        return self.tokenize_text(text)


class YtMultiTableDatasetIterator:
    def __init__(self,
                 data_sources: List[DataSource],
                 batch_size: int,
                 block_size: int,
                 row_parser: SampleParser,
                 max_num_words: int,
                 words_shift: int,
                 max_num_tokens: int,
                 empty_samples_proportion: float,
                 reader_id: int,
                 total_readers: int,
                 pad_to: int,
                 sort_by_length: bool,
                 buffer_size: int = 1000):
        self._data_sources = data_sources
        self._batch_size = batch_size
        self._block_size = block_size
        self._row_parser = row_parser
        self._max_num_words = max_num_words
        self._words_shift = words_shift
        self._max_num_tokens = max_num_tokens
        self._pad_to = pad_to
        self._sort_by_length = sort_by_length
        self._empty_samples_proportion = empty_samples_proportion
        self._buffer_size = buffer_size

        self._buffer = []
        self._multitable_reader = MultiTableYtReader(data_sources, reader_id, total_readers)

    def set_progress(self, thread_state: ReadThreadState):
        self._multitable_reader.set_progress(thread_state)

    def __iter__(self):
        return self

    def __next__(self):
        samples = []
        while len(samples) < self._batch_size * self._block_size:
            self._update_buffer()
            text = self._buffer.pop() if np.random.uniform() > self._empty_samples_proportion else ""
            try:
                sample = self._row_parser.tokenize_text(text)
                if sample[2].item() > self._max_num_tokens:
                    sys.stderr.write(f"Skipping big sample {sample[2].item()}\n")
                else:
                    samples.append(sample)
            except Exception as e:
                sys.stderr.write(f"Error parsing sample {text}: {e}\n")

        if self._sort_by_length:
            raise Exception('блять)')

        blocks = [TextBatch.create_from_list(samples[i:i + self._batch_size], self._pad_to)
                  for i in range(0, len(samples), self._batch_size)]
        reader_state = self._multitable_reader.get_progress()

        return BatchBlock(blocks, reader_state)

    def _crop_text(self, text):
        splitted = text.split(" ")
        if len(splitted) > self._max_num_words:
            return [" ".join(splitted[i:i + self._max_num_words])
                    for i in range(0, len(splitted) - self._max_num_words, self._words_shift)]
        return [text]

    def _update_buffer(self):
        if len(self._buffer):
            return
        while len(self._buffer) < self._buffer_size:
            yt_row, _ = next(self._multitable_reader)
            text = self._row_parser.get_text(yt_row)
            if not text:
                continue
            self._buffer.extend(self._crop_text(text))
        random.shuffle(self._buffer)


class YtMultiTableDataset(IterableDataset):
    def __init__(self,
                 data_sources: List[DataSource],
                 dictionary: Dictionary,
                 batch_size: int,
                 block_size: int,
                 max_num_words: int,
                 words_shift: int,
                 max_num_tokens: int,
                 empty_samples_proportion: float,
                 latin_policy: LatinPolicy,
                 text_processor: Optional[SentenceProcessor],
                 use_bos: bool,
                 use_eos: bool,
                 pad_to: int,
                 sort_by_length: bool):
        self._data_sources = data_sources
        self._batch_size = batch_size
        self._block_size = block_size
        self._max_num_words = max_num_words
        self._words_shift = words_shift
        self._max_num_tokens = max_num_tokens
        self._empty_samples_proportion = empty_samples_proportion
        self._pad_to = pad_to
        self._sort_by_length = sort_by_length
        self._readers_checkpoint = None

        self._row_parser = SampleParser(dictionary, latin_policy, text_processor, use_bos, use_eos)

    def set_progress(self, progress: ReaderProgress):
        self._readers_checkpoint = copy.deepcopy(progress)

    def __iter__(self):
        reader_id, total_readers = readers_info()
        iterator = YtMultiTableDatasetIterator(self._data_sources,
                                               self._batch_size,
                                               self._block_size,
                                               self._row_parser,
                                               self._max_num_words,
                                               self._words_shift,
                                               self._max_num_tokens,
                                               self._empty_samples_proportion,
                                               reader_id,
                                               total_readers,
                                               self._pad_to,
                                               self._sort_by_length)
        if self._readers_checkpoint is not None:
            iterator.set_progress(self._readers_checkpoint.state(reader_id))
        return iterator

def identity(x):
    return x


class YtTrainDataLayer:
    def __init__(self,
                 tables: List[str],
                 batch_size: int,
                 block_size: int,
                 dictionary: Dictionary,
                 max_num_words: int,
                 words_shift: int,
                 max_num_tokens: int,
                 empty_samples_proportion: float,
                 latin_policy: LatinPolicy,
                 text_processor: Optional[SentenceProcessor],
                 use_bos: bool,
                 use_eos: bool,
                 sort_by_length: bool,
                 pad_to: int,
                 read_threads: int):
        self._tables = [YtTableInfo(data_path) for data_path in tables]
        self._batch_size = batch_size
        self._dataset = YtMultiTableDataset(data_sources=[DataSource(source) for source in self._tables],
                                            dictionary=dictionary,
                                            batch_size=batch_size,
                                            block_size=block_size,
                                            max_num_words=max_num_words,
                                            words_shift=words_shift,
                                            max_num_tokens=max_num_tokens,
                                            empty_samples_proportion=empty_samples_proportion,
                                            latin_policy=latin_policy,
                                            text_processor=text_processor,
                                            use_bos=use_bos,
                                            use_eos=use_eos,
                                            pad_to=pad_to,
                                            sort_by_length=sort_by_length)
        self._data_loader = torch.utils.data.DataLoader(self._dataset,
                                                        batch_size=1,
                                                        collate_fn=identity,
                                                        num_workers=read_threads,
                                                        pin_memory=True)

    def set_progress(self, state: ReaderProgress):
        self._dataset.set_progress(state)

    def data_sources(self):
        return self._tables

    @property
    def data_loader(self):
        return self._data_loader


class YtTestDataset:
    def __init__(self,
                 path: str,
                 dictionary: Dictionary,
                 batch_size: int,
                 latin_policy: LatinPolicy,
                 text_processor: Optional[SentenceProcessor],
                 use_bos: bool,
                 use_eos: bool,
                 sort_by_duration: bool,
                 in_memory: bool,
                 pad_to: int,
                 read_threads: int):
        self._table_info = YtTableInfo(path)
        self._batch_size = batch_size
        self._pad_to = pad_to
        self._in_memory = in_memory

        whole_table_reader = YTTableParallelReader(self._table_info.cluster, self._table_info.table,
                                                   cache_size=128, num_readers=read_threads)
        self._total_samples = whole_table_reader.num_rows

        self._row_parser = SampleParser(dictionary, latin_policy, text_processor, use_bos, use_eos)

        num_workers = 1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size()
        my_rank = 0 if not torch.distributed.is_initialized() else torch.distributed.get_rank()
        total_rows = whole_table_reader.num_rows
        total_rows = (total_rows // num_workers) * num_workers
        rows_per_worker = (total_rows // num_workers)
        offset = rows_per_worker * my_rank
        next_offset = rows_per_worker * (my_rank + 1)
        self._rows_per_worker = rows_per_worker
        self._reader = whole_table_reader.make_subset_reader(offset, next_offset)

        if in_memory:
            timer = Timer()
            timer.reset()
            self._load_batches(sort_by_duration)
            LOG.debug(f"Batches on GPU #{my_rank}: {len(self._batches)}")
            LOG.debug(f"Read {rows_per_worker} samples in {timer.passed()} seconds")
        else:
            LOG.debug(f"Number of expected rows to read: {rows_per_worker}")

    def _load_batches(self, sort_by_duration):
        samples = [self._row_parser.parse(sample) for sample in self._reader]

        if sort_by_duration:
            raise Exception('блять)')

        self._batches = [TextBatch.create_from_list(samples[i:i + self._batch_size], pad_to=self._pad_to).pin_memory()
                         for i in range(0, len(samples), self._batch_size)]

    def _in_memory_stream(self):
        for batch in self._batches:
            yield batch.clone_to_gpu()

    def _lazy_stream(self):
        chunks = map(partial(filter, None), zip_longest(*repeat(iter(self._reader), self._batch_size)))
        for chunk in chunks:
            chunk = list(map(self._row_parser.parse, chunk))
            batch = TextBatch.create_from_list(chunk, pad_to=self._pad_to).pin_memory()
            yield batch.clone_to_gpu()

    def __iter__(self):
        return self._in_memory_stream() if self._in_memory else self._lazy_stream()

    @property
    def total_batches(self):
        return len(self._batches) if self._in_memory else math.ceil(self._rows_per_worker / self._batch_size)

    @property
    def total_samples(self):
        return self._total_samples
