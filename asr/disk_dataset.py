import copy
from functools import partial
from itertools import zip_longest
import math
from typing import Dict, Optional

from torch import Tensor
import torch.distributed as dist
from torch.utils.data import Dataset, IterableDataset

from asr.features_extractor import FeatureExtractorFactory
from asr.spectogram_augmentations import SpectrogramAugmentatator

from common.dictionary import Dictionary
from common.disk_utils import *
from common.utils import Timer
from common.text_processor import SentenceProcessor

LOG = logging.getLogger()


class SpeechBatch:
    @staticmethod
    def create_from_list(batch: List, frame_shift: float, pad_to: int):
#         LOG.info(f'Will create from batch: {len(batch)}')
        batch_size = len(batch)

        features_dim = batch[0][0].size(1)
        max_features_length = ((find_max_len(batch, 0) + pad_to - 1) // pad_to) * pad_to

        speakers_num = len(batch[0][3])
        max_tokens_lengths = [max(lens[sp].item() for _, _, _, lens in batch) for sp in range(speakers_num)]

        batched_features = torch.zeros(batch_size, max_features_length, features_dim)
        batched_tokens = [torch.zeros(batch_size, max_tokens_len) for max_tokens_len in max_tokens_lengths]
        
        features_lengths = []
        tokens_lengths = [[] for _ in range(speakers_num)]

        total_seconds = 0
        for i, sample in enumerate(batch):
            features, features_length, tokens_list, tokens_lens_list = sample

            batched_features[i].narrow(0, 0, features_length.item()).copy_(features)
            features_lengths.append(features_length)

            for s in range(speakers_num):
                batched_tokens[s][i].narrow(0, 0, tokens_lens_list[s].item()).copy_(tokens_list[s])
                tokens_lengths[s].append(tokens_lens_list[s])

            total_seconds += features_length.item() * frame_shift / 1000

        features = batched_features.permute([0, 2, 1])
        features_lengths = torch.stack(features_lengths)
        tokens = batched_tokens
        tokens_lengths = [torch.stack(tokens_lens) for tokens_lens in tokens_lengths]
        return SpeechBatch(features, features_lengths, tokens, tokens_lengths, total_seconds)

    def __init__(self, features: Tensor, features_lengths: Tensor,
                 tokens: List[Tensor], tokens_lengths: List[Tensor],
                 total_seconds: float):
        self._features = features
        self._features_lengths = features_lengths
        self._speakers_num = len(tokens)
        self.tokens = [t.long() for t in tokens]
        self.tokens_lengths = [t_lens.long() for t_lens in tokens_lengths]
        self._total_seconds = total_seconds
        self._batch_size = self._features.size(0)

    def cuda(self):
        self._features = self._features.cuda()
        self._features_lengths = self._features_lengths.cuda()
        for s in range(self._speakers_num):
            if self.tokens[s] is not None:
                self.tokens[s] = self.tokens[s].cuda()
            if self.tokens_lengths[s] is not None:
                self.tokens_lengths[s] = self.tokens_lengths[s].cuda()
        return self

    def pin_memory(self):
        self._features = self._features.pin_memory()
        self._features_lengths = self._features_lengths.pin_memory()
        for s in range(self._speakers_num):
            self.tokens[s] = self.tokens[s].pin_memory()
            self.tokens_lengths[s] = self.tokens_lengths[s].pin_memory()
        return self

    def __len__(self):
        return self._batch_size

    @property
    def features(self):
        return self._features

    @property
    def features_lengths(self):
        return self._features_lengths

    @property
    def total_seconds(self):
        return self._total_seconds

    def clone_to_gpu(self):
        return SpeechBatch(clone_tensor_to_cuda(self._features),
                           clone_tensor_to_cuda(self._features_lengths),
                           [clone_tensor_to_cuda(t) for t in self.tokens],
                           [clone_tensor_to_cuda(t_len) for t_len in self.tokens_lengths],
                           self._total_seconds)


class SampleParser:
    def __init__(self, dictionary: Dictionary, wave_augmentator, spec_augmentator, features_config: dict,
                 data_type: str, latin_policy: LatinPolicy, text_processor: Optional[SentenceProcessor]):
        self._dictionary = dictionary
        self._features_config = features_config
        self._features_extractor = FeatureExtractorFactory.create(data_type, features_config, wave_augmentator,
                                                                  spec_augmentator)
        self._latin_policy = latin_policy
        self._text_processor = text_processor

    def parse(self, record: dict):
        features = torch.tensor(self._features_extractor.extract(record))
        text = load_text_from_record(record, self._latin_policy)
        if self._text_processor is not None:
            text = self._text_processor.process_sentence(text)
        tokens = self._dictionary.encode(text)
        if not len(tokens):
            tokens = self._dictionary.encode(" ")
            assert len(tokens) > 0

        num_frames = features.shape[0]
        tokens_len = len(tokens)

        return features, torch.tensor(num_frames).int(), torch.tensor(tokens), torch.tensor(tokens_len).int()


def take_first(sample):
    return sample[1]


class DiskMultiDirsDatasetIterator:
    def __init__(self, directories: List[str], row_parsers: Dict[str, SampleParser], batch_size: int,
                 block_size: int, max_duration_frames: int, frame_shift: int, reader_id: int, total_readers: int,
                 pad_to: int, sort_by_length: bool, seed=None):
        # LOG.debug(f"DiskMultiDirsDatasetIterator id={gpu_id()} seed={seed}")
        self._directories = directories
        self._row_parsers = row_parsers
        self._batch_size = batch_size
        self._block_size = block_size
        self._max_duration_frames = max_duration_frames
        self._frame_shift = frame_shift
        self._pad_to = pad_to
        self._sort_by_length = sort_by_length
        self._current_dir_iterator = None
        self._current_dir_index = 0
        self._multidir_reader = MultiDirsDiskReader(directories, reader_id, total_readers, seed=seed)

    def set_progress(self, thread_state: ReadThreadState):
        self._multidir_reader.set_progress(thread_state)

    def __iter__(self):
        LOG.debug("iterator iter")
        # sys.stderr.write(f"Iterator iter\n")
        return self

    def __next__(self):
        LOG.debug("iterator next")
        # sys.stderr.write(f"Iterator iter\n")
        samples = []
        while len(samples) < self._batch_size * self._block_size:
            record, data_type = next(self._multidir_reader)
            # LOG.debug(f"id={gpu_id() if torch.distributed.is_initialized() else 0} record_text={record['text']}")

            try:
                sample = self._row_parsers[data_type].parse(record)
                if self._max_duration_frames is not None and sample[0].shape[0] >= self._max_duration_frames:
                    sys.stderr.write("Skipping big sample {}\n".format(sample[0].shape[0]))
                else:
                    samples.append(sample)
            except Exception as e:
                sys.stderr.write(f"Error parsing sample {record['text']} : {e}\n")
        if self._sort_by_length:
            samples = sorted(samples, key=take_first)
        blocks = [SpeechBatch.create_from_list(samples[i:i + self._batch_size], self._frame_shift, self._pad_to)
                  for i in range(0, len(samples), self._batch_size)]
        reader_state = self._multidir_reader.get_progress()

        batch_block = BatchBlock(blocks, reader_state)
        LOG.debug(f'batch_features_shape = {batch_block.batches[0].features.shape}')

        return batch_block


def identity(x):
    return x


class DiskMultiDirsDataset(IterableDataset):
    def __init__(self,
                 directories: List[str],
                 dictionary: Dictionary,
                 features_config: dict,
                 batch_size: int,
                 block_size: int,
                 max_duration: int,
                 wave_augmentation_config:
                 dict, spec_augmentation_config: dict,
                 latin_policy: LatinPolicy,
                 text_processor: Optional[SentenceProcessor],
                 pad_to: int,
                 sort_by_length: bool,
                 seed=None):
        # LOG.debug(f"DiskMultiDirsDataset id={gpu_id()} seed={seed}")
        self._seed = seed

        self._directories = directories
        self._batch_size = batch_size
        self._block_size = block_size
        self._max_duration_frames = max_duration * 2000.0 / features_config["frame-shift"]
        self._frame_shift = features_config["frame-shift"]
        self._pad_to = pad_to
        self._sort_by_length = sort_by_length
        self._readers_checkpoint = None

        self._wave_augmentator = identity
        self._spec_augmentator = identity
        if wave_augmentation_config is not None and len(wave_augmentation_config) > 0:
            from .wave_augmentations import WaveAugmentor
            self._wave_augmentator = WaveAugmentor.from_config(augmentation_config=wave_augmentation_config,
                                                               features_config=features_config,
                                                               max_duration=max_duration)
        if spec_augmentation_config is not None and len(spec_augmentation_config) > 0:
            self._spec_augmentator = SpectrogramAugmentatator.from_config(augmentation_config=spec_augmentation_config)

        self._parsers = dict()

        data_type = "disk-raw"
        self._parsers[data_type] = SampleParser(dictionary, self._wave_augmentator, self._spec_augmentator,
                                                features_config, data_type, latin_policy, text_processor)

    def set_progress(self, progress: ReaderProgress):
        self._readers_checkpoint = copy.deepcopy(progress)

    def __iter__(self):
        reader_id, total_readers = self._readers_info()
        LOG.debug(f"Multi dirs dataset iter")
        # sys.stderr.write(f"Multi dirs dataset iter\n")
        iterator = DiskMultiDirsDatasetIterator(self._directories, self._parsers, self._batch_size, self._block_size,
                                                self._max_duration_frames, self._frame_shift, reader_id, total_readers,
                                                self._pad_to, self._sort_by_length, seed=self._seed)
        if self._readers_checkpoint is not None:
            iterator.set_progress(self._readers_checkpoint.state(reader_id))
        return iterator

    def _readers_info(self):
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


class DiskTrainDataLayer:
    # TODO: 'speech_batch_block in train_data_layer._data_loader';
    #  speech_batch_block : (SpeechBatch, SpeechBatch)
    def __init__(self,
                 directories: List[str],
                 batch_size: int,
                 block_size: int,
                 dictionary: Dictionary,
                 features_config: dict,
                 max_duration: int,
                 wave_augmentation_config: dict,
                 spec_augmentation_config: dict,
                 latin_policy: LatinPolicy,
                 text_processor: Optional[SentenceProcessor],
                 sort_by_length: bool,
                 pad_to: int,
                 read_threads: int,
                 seed=None):
        # LOG.debug(f"DiskTrainDataLayer id={gpu_id()} seed={seed}")
        self._directories = directories
        self._batch_size = batch_size
        self._dataset = DiskMultiDirsDataset(directories=directories,
                                             dictionary=dictionary,
                                             features_config=features_config,
                                             batch_size=batch_size,
                                             block_size=block_size,
                                             max_duration=max_duration,
                                             wave_augmentation_config=wave_augmentation_config,
                                             spec_augmentation_config=spec_augmentation_config,
                                             latin_policy=latin_policy,
                                             text_processor=text_processor,
                                             pad_to=pad_to,
                                             sort_by_length=sort_by_length,
                                             seed=seed)
        self._data_loader = torch.utils.data.DataLoader(self._dataset,
                                                        batch_size=1,
                                                        collate_fn=identity,
                                                        num_workers=read_threads,
                                                        pin_memory=True)

    def set_progress(self, state: ReaderProgress):
        pass
        # self._dataset.set_progress(state)

    def data_sources(self):
        return self._directories

    @property
    def data_loader(self):
        return self._data_loader


def take_first_item(sample):
    return sample[1].item()


class DiskTestDataset:
    def __init__(self,
                 directory: str,
                 dictionary: Dictionary,
                 batch_size: int,
                 features_config: dict,
                 latin_policy: LatinPolicy,
                 text_processor: Optional[SentenceProcessor],
                 sort_by_duration: bool,
                 in_memory: bool,
                 pad_to: int,
                 read_threads: int):
        self._directory = directory
        self._batch_size = batch_size
        self._frame_shift = features_config["frame-shift"]
        self._pad_to = pad_to
        self._in_memory = in_memory

        whole_table_reader = DiskDirReader(self._directory, 0)
        self._total_samples = get_dir_samples_number(self._directory)

        self._features_extractor = FeatureExtractorFactory.create("disk-raw", features_config,
                                                                  identity, identity)
        self._parser = SampleParser(dictionary=dictionary,
                                    wave_augmentator=identity,
                                    spec_augmentator=identity,
                                    features_config=features_config,
                                    data_type="disk-raw",
                                    latin_policy=latin_policy,
                                    text_processor=text_processor)

        num_workers = 1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size()
        my_rank = 0 if not torch.distributed.is_initialized() else torch.distributed.get_rank()
        total_rows = self._total_samples
        total_rows = (total_rows // num_workers) * num_workers
        rows_per_worker = (total_rows // num_workers)
        offset = rows_per_worker * my_rank
        next_offset = rows_per_worker * (my_rank + 1)
        self._rows_per_worker = rows_per_worker
        self._reader = whole_table_reader.make_subset_reader(offset, total_rows - next_offset)

        if in_memory:
            timer = Timer()
            timer.reset()
            self._load_batches(sort_by_duration)
            LOG.debug(f"Batches on GPU #{my_rank}: {len(self._batches)}")
            LOG.debug(f"Read {rows_per_worker} samples in {timer.passed()} seconds")
        else:
            LOG.debug(f"Number of expected rows to read: {rows_per_worker}")

    def _load_batches(self, sort_by_duration):
        samples = [self._parser.parse(sample) for sample, _ in self._reader]

        if sort_by_duration:
            samples.sort(key=take_first_item)

        self._batches = [SpeechBatch.create_from_list(samples[i:i + self._batch_size], frame_shift=self._frame_shift,
                                                      pad_to=self._pad_to).pin_memory()
                         for i in range(0, len(samples), self._batch_size)]

    def _in_memory_stream(self):
        for batch in self._batches:
            yield batch.clone_to_gpu()

    def _lazy_stream(self):
        chunks = map(partial(filter, None), zip_longest(*repeat(iter(self._reader), self._batch_size)))
        for chunk in chunks:
            chunk = list(map(self._parser.parse, chunk))
            batch = SpeechBatch.create_from_list(chunk, frame_shift=self._frame_shift, pad_to=self._pad_to).pin_memory()
            yield batch.clone_to_gpu()

    def __iter__(self):
        return self._in_memory_stream() if self._in_memory else self._lazy_stream()

    @property
    def total_batches(self):
        return len(self._batches) if self._in_memory else math.ceil(self._rows_per_worker / self._batch_size)

    @property
    def total_samples(self):
        return self._total_samples
