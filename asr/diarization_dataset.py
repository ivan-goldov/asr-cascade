import copy
from typing import Optional

from scipy.io import wavfile
import wget

import warnings
from torch.utils.data import Dataset, IterableDataset

from asr.features_extractor import FeatureExtractorFactory
from asr.spectogram_augmentations import SpectrogramAugmentatator

from common.dictionary import Dictionary
from common.text_processor import SentenceProcessor
from common.diarization_utils import *
from asr.disk_dataset import SpeechBatch
from common.synthetic_generation_utils import mix_audios, normalize

LOG = logging.getLogger()


def parse_from_audios(audios, texts, rate, features_extractor, text_processor, dictionary):
    record = {'sample_rate': rate, 'data': audios}
    parsed_record = features_extractor.extract(record)
    if parsed_record is None:
        LOG.info('Failed to mix speakers, skipping sample...')
        return None
    features = torch.tensor(parsed_record, dtype=torch.float32)
    num_frames = features.shape[0]

    tokens_list = []
    tokens_len_list = []
    for text in texts:
        if text_processor is not None:
            text = text_processor.process_sentence(text)
        tokens = dictionary.encode(text)
        if not len(tokens):
            tokens = dictionary.encode(" ")
            assert len(tokens) > 0
        tokens_list.append(tokens)
        tokens_len_list.append(len(tokens))
    return features, torch.tensor(num_frames).int(), [torch.tensor(tokens) for tokens in tokens_list], \
           [torch.tensor(tokens_len) for tokens_len in tokens_len_list]


class DiskSampleParser:
    def __init__(self, dictionary: Dictionary, wave_augmentator, spec_augmentator, features_config: dict,
                 data_type: str, latin_policy: LatinPolicy, text_processor: Optional[SentenceProcessor],
                 max_speakers_num: int, speakers_num_frequency: List[float], constant_gap: Optional[float]):
        self._data_dir = 'data/disk_data'
        self._dictionary = dictionary
        self._features_config = features_config
        LOG.info('data type: {}'.format(data_type))
        self._features_extractor = FeatureExtractorFactory.create(data_type, features_config, wave_augmentator,
                                                                  spec_augmentator)
        self._latin_policy = latin_policy
        self._text_processor = text_processor

        self._max_speakers_num = max_speakers_num
        self._speakers_num_frequency = speakers_num_frequency
        self._constant_gap = constant_gap

    def _get_inp_filepath(self, record_id):
        return os.path.join(os.getcwd(), self._data_dir, record_id + '.wav')

    def _download_wav(self, link, rec_id):
        filepath = self._get_inp_filepath(rec_id)
        if not os.path.exists(filepath):
            destination = os.path.join(os.getcwd(), self._data_dir)
            wget.download(link, out=destination)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rate, s = wavfile.read(filepath)
        return rate, s

    def parse_audios(self, rows: List[dict]):
        texts, audios = [], []
        sample_speakers_num = np.random.choice(
            np.arange(1, self._max_speakers_num + 1), 1, p=self._speakers_num_frequency)[0]
        for row in rows[:sample_speakers_num]:
            rate, s = self._download_wav(row['link'], row['record_id'])
            assert rate == self._features_config['sample-frequency']
            texts.append(row['toloka_text'])
            audios.append(s)
        for _ in range(self._max_speakers_num - sample_speakers_num):
            texts.append('')
        
        if sorted([-min(1, len(a)) for a in texts]) != [-min(1, len(a)) for a in texts]:
            return None, []
        mixed_audio = mix_audios(audios, self._features_config['sample-frequency'], constant_gap=self._constant_gap)
        return mixed_audio, texts

    def parse_from_audios(self, audios, texts):
        return parse_from_audios(audios, texts, self._features_config['sample-frequency'],
                                 self._features_extractor, self._text_processor, self._dictionary)

    def parse(self, rows: List[dict]):
        audios, texts = self.parse_audios(rows)
        return self.parse_from_audios(audios, texts)


class KekosParser:
    def __init__(self, 
                 dictionary: Dictionary, wave_augmentator, spec_augmentator, features_config: dict,
                 data_type: str, latin_policy: LatinPolicy, text_processor: Optional[SentenceProcessor], **kwargs):
        self._dictionary = dictionary
        self._features_config = features_config
        self._features_extractor = FeatureExtractorFactory.create("example", features_config, wave_augmentator,
                                                                  spec_augmentator)
        self._text_processor = text_processor

        self._s3_loader = S3Loader()

    def parse_audios(self, s: List[Dict]):
        sample = s
        if isinstance(s, list):
            sample = s[0]
        texts = [sample['text1'], sample['text2']]
        audio1 = np.array(self._s3_loader.load(
            1, sample['channel_count'], sample['sample_rate_hertz'],
            sample['audio_s3_obj_bucket'],
            sample['audio_s3_obj_key1'],
            16000))
        audio2 = np.array(self._s3_loader.load(
            1, sample['channel_count'], sample['sample_rate_hertz'],
            sample['audio_s3_obj_bucket'],
            sample['audio_s3_obj_key2'],
            16000))
        if len(audio1) < len(audio2) and len(audio2) - len(audio1) < 5:
            audio1 = np.concatenate((audio1, np.zeros((len(audio2) - len(audio1),))))
        if len(audio1) > len(audio2) and len(audio1) - len(audio2) < 5:
            audio2 = np.concatenate((audio2, np.zeros((len(audio1) - len(audio2),))))
        audio = normalize(audio1 + audio2)

        if self._text_processor is not None:
            for i in range(len(texts)):
                texts[i] = self._text_processor.process_sentence(texts[i])
        
        return audio, texts

    def parse_from_audios(self, audio, texts):
        return parse_from_audios(audio, texts, self._features_config['sample-frequency'],
                                 self._features_extractor, self._text_processor, self._dictionary)

    def parse(self, row: Dict):
        audio, texts = self.parse_audios(row)
        print('БЛЯТЬ1')
        return self.parse_from_audios(audio, texts)


class YtSampleParser:
    def __init__(self, dictionary: Dictionary, wave_augmentator, spec_augmentator, features_config: dict,
                 data_type: str, latin_policy: LatinPolicy, text_processor: Optional[SentenceProcessor],
                 max_speakers_num: int, speakers_num_frequency: List[float], constant_gap: Optional[float]):
        self._dictionary = dictionary
        self._features_config = features_config
        LOG.info('data type: {}'.format(data_type))
        self._features_extractor = FeatureExtractorFactory.create(data_type, features_config, wave_augmentator,
                                                                  spec_augmentator)
        self._latin_policy = latin_policy
        self._text_processor = text_processor

        self._max_speakers_num = max_speakers_num
        self._speakers_num_frequency = speakers_num_frequency
        self._constant_gap = constant_gap

        self._s3_loader = S3Loader()

    def parse_audios(self, yt_rows: List[dict]):
        texts, audios = [], []
        sample_speakers_num = np.random.choice(
            np.arange(1, self._max_speakers_num + 1), 1, p=self._speakers_num_frequency)[0]
        for yt_row in yt_rows[:sample_speakers_num]:
            texts.append(load_text_from_yt_row(yt_row, self._latin_policy))
            audios.append(self._s3_loader.load_from_yt_row(yt_row, self._features_config['sample-frequency']))
        for _ in range(self._max_speakers_num - sample_speakers_num):
            texts.append('')
        mixed_audio = mix_audios(audios, self._features_config['sample-frequency'], constant_gap=self._constant_gap)
        return mixed_audio, texts

    def parse_from_audios(self, audios, texts):
        return parse_from_audios(audios, texts, self._features_config['sample-frequency'],
                                 self._features_extractor, self._text_processor, self._dictionary)

    def parse(self, yt_rows: List[dict]):
        audios, texts = self.parse_audios(yt_rows)
        return self.parse_from_audios(audios, texts)


class MultiTableDatasetIterator:
    def __init__(self, data_sources: List, row_parsers: Dict,
                 batch_size: int, block_size: int, max_duration_frames: int, frame_shift: int,
                 reader_id: int, total_readers: int, pad_to: int, sort_by_length: bool,
                 merge_short_records: bool, dictionary: Dictionary, max_speakers_num: int,
                 max_sz: Optional[int] = None):
        self._data_sources = data_sources
        self._row_parsers = row_parsers
        self._batch_size = batch_size
        self._block_size = block_size
        self._max_duration_frames = max_duration_frames
        self._frame_shift = frame_shift
        self._pad_to = pad_to
        self._sort_by_length = sort_by_length
        self._merge_short_records = merge_short_records
        self._multitable_reader = MultiTableYtReader(
            data_sources, reader_id, total_readers, max_speakers_num)
        self._dictionary = dictionary
        self._max_sz = max_sz

    def set_progress(self, thread_state: ReadThreadState):
        self._multitable_reader.set_progress(thread_state)

    def __iter__(self):
        return self

    def _get_sample(self):
        rows, data_type = next(self._multitable_reader)
        try:
            mixed_audio, texts = self._row_parsers[data_type].parse_audios(rows)
            sample = self._row_parsers[data_type].parse_from_audios(mixed_audio, texts)
        except Exception as e:
            sys.stderr.write(str(e))
            return self._get_sample()
        if sample is None:
            return self._get_sample()
        return rows, data_type, sample, mixed_audio, texts

    def next_single_audio(self):
        rows, data_type, sample, mixed_audio, texts = self._get_sample()
        batch_block = BatchBlock([SpeechBatch.create_from_list([sample], self._frame_shift, self._pad_to)], None)
        return mixed_audio, texts, batch_block

    def __next__(self):
        if self._max_sz is not None:
            if self._max_sz <= 0:
                raise StopIteration
            self._max_sz -= 1

        samples = []
        while len(samples) < self._batch_size * self._block_size:
            try:
                rows, data_type, sample, _, _ = self._get_sample()

                if self._max_duration_frames is not None and sample[0].shape[0] >= self._max_duration_frames:
                    LOG.debug("Skipping big sample {} >= {}\n".format(sample[0].shape[0], self._max_duration_frames))
                else:
                    if self._merge_short_records and len(samples) > 0:
                        raise Exception('блять)')
                    samples.append(sample)
            except Exception as e:
                sys.stderr.write(str(e))
                sample_texts = [row[b'text'].decode("utf-8") for row in rows]
                sys.stderr.write(f"Error parsing sample {', '.join(sample_texts)} : {e}\n")
        if self._sort_by_length:
            raise Exception('блять)')
        blocks = [SpeechBatch.create_from_list(samples[i:i + self._batch_size], self._frame_shift, self._pad_to)
                  for i in range(0, len(samples), self._batch_size)]
        reader_state = self._multitable_reader.get_progress()

        return BatchBlock(blocks, reader_state)


class MultiTableDataset(IterableDataset):
    def __init__(self,
                 tables: List,
                 dictionary: Dictionary,
                 features_config: dict,
                 batch_size: int,
                 block_size: int,
                 max_duration: int,
                 wave_augmentation_config: dict,
                 spec_augmentation_config: dict,
                 latin_policy: LatinPolicy,
                 text_processor,
                 pad_to: int,
                 sort_by_length: bool,
                 merge_short_records: bool,
                 max_speakers_num: int,
                 speakers_num_frequency: List[float],
                 constant_gap: Optional[float] = None):
        self._data_sources = [create_datasource(t, weight, start, end) for t, weight, (start, end) in tables]
        self._batch_size = batch_size
        self._block_size = block_size
        self._max_duration_frames = max_duration * 1000.0 / features_config["frame-shift"]
        self._frame_shift = features_config["frame-shift"]
        self._pad_to = pad_to
        self._sort_by_length = sort_by_length
        self._merge_short_records = merge_short_records
        self._readers_checkpoint = None
        self._dictionary = dictionary
        self._max_speakers_num = max_speakers_num

        self._wave_augmentator = identity
        self._spec_augmentator = identity
        if wave_augmentation_config is not None and len(wave_augmentation_config) > 0:
            from .wave_augmentations import WaveAugmentor
            self._wave_augmentator = WaveAugmentor.from_config(augmentation_config=wave_augmentation_config,
                                                               features_config=features_config,
                                                               max_duration=max_duration)
        if spec_augmentation_config is not None and len(spec_augmentation_config) > 0:
            self._spec_augmentator = SpectrogramAugmentatator.from_config(augmentation_config=spec_augmentation_config)

        self._data_types = {data_source.data_type for data_source in self._data_sources}
        self._parsers = dict()
        for data_type in self._data_types:
            _sample_parser = {'disk-raw': DiskSampleParser, 'yt-raw': YtSampleParser, 'kekos': KekosParser}
            self._parsers[data_type] = _sample_parser[data_type](
                dictionary, self._wave_augmentator, self._spec_augmentator,
                features_config, data_type, latin_policy, text_processor,
                max_speakers_num=max_speakers_num,
                speakers_num_frequency=speakers_num_frequency,
                constant_gap=constant_gap)

    def set_progress(self, progress: ReaderProgress):
        self._readers_checkpoint = copy.deepcopy(progress)

    def __iter__(self, max_sz=None):
        reader_id, total_readers = self._readers_info()
        iterator = MultiTableDatasetIterator(self._data_sources, self._parsers, self._batch_size, self._block_size,
                                             self._max_duration_frames, self._frame_shift, reader_id, total_readers,
                                             self._pad_to, self._sort_by_length, self._merge_short_records,
                                             self._dictionary, self._max_speakers_num, max_sz)

        if self._readers_checkpoint is not None:
            iterator.set_progress(self._readers_checkpoint.state(reader_id))
        return iterator

    def _readers_info(self):
        data_load_process_info = torch.utils.data.get_worker_info()
        if data_load_process_info is not None:
            return data_load_process_info.id, data_load_process_info.num_workers
        else:
            return 0, 1


def identity(x):
    return x


class TrainDataLayer:
    def __init__(self,
                 tables: List[Tuple[str, float, Tuple[float, float]]],
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
                 merge_short_records: bool,
                 pad_to: int,
                 read_threads: int,
                 max_speakers_num: int,
                 speakers_num_frequency: List[float],
                 overlap: Optional[float]):
        assert len(speakers_num_frequency) == max_speakers_num
        assert sum(speakers_num_frequency) == 1.0

        self._batch_size = batch_size
        self._dataset = MultiTableDataset(tables=tables,
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
                                          merge_short_records=merge_short_records,
                                          max_speakers_num=max_speakers_num,
                                          speakers_num_frequency=speakers_num_frequency,
                                          constant_gap=overlap)
        self._data_loader = torch.utils.data.DataLoader(self._dataset,
                                                        batch_size=1,
                                                        collate_fn=identity,
                                                        num_workers=read_threads,
                                                        pin_memory=True)

    def set_progress(self, state: ReaderProgress):
        self._dataset.set_progress(state)

    def data_sources(self):
        return self._dataset._data_sources

    @property
    def data_loader(self):
        return self._data_loader


def take_first_item(sample):
    return sample[1].item()
