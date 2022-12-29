from typing import Optional
from scipy.io import wavfile

from asr.features_extractor import FeatureExtractorFactory
from asr.spectogram_augmentations import SpectrogramAugmentatator

from common.dictionary import Dictionary
from common.text_processor import SentenceProcessor
from common.diarization_utils import *
from asr.disk_dataset import SpeechBatch
from asr.diarization_dataset import parse_from_audios, identity, take_first_item, KekosParser
from asr.diarization_dataset import S3Loader, normalize


class ExampleParser:
    def __init__(self, dictionary: Dictionary, wave_augmentator, spec_augmentator, features_config: dict,
                 text_processor: Optional[SentenceProcessor]):
        self._data_dir = 'data/validation_data/lectures'
        self._dictionary = dictionary
        self._features_config = features_config
        self._features_extractor = FeatureExtractorFactory.create("example", features_config, wave_augmentator,
                                                                  spec_augmentator)
        self._text_processor = text_processor

    def parse_audios(self, sample: Dict):
        audio_filename, texts = sample
        rate, audio = wavfile.read(os.path.join(self._data_dir, audio_filename))
        assert rate == self._features_config['sample-frequency']
        audio = audio.mean(axis=1)  # 2 streams
        audio = normalize(audio)

        if self._text_processor is not None:
            for i in range(len(texts)):
                texts[i] = self._text_processor.process_sentence(texts[i])

        return audio, texts

    def parse_from_audios(self, audio, texts):
        return parse_from_audios(audio, texts, self._features_config['sample-frequency'],
                                 self._features_extractor, self._text_processor, self._dictionary)

    def parse(self, rows: Dict):
        audio, texts = self.parse_audios(rows)
        return self.parse_from_audios(audio, texts)


class ExampleTestDataset:
    def __init__(self,
                 data_filepath: str,
                 dictionary: Dictionary,
                 batch_size: int,
                 features_config: dict,
                 text_processor,
                 sort_by_length: bool,
                 pad_to: int,
                 **kwargs):
        self._data_filepath = data_filepath
        self._batch_size = batch_size
        self._frame_shift = features_config["frame-shift"]
        self._pad_to = pad_to

        self._wave_augmentator = identity
        self._spec_augmentator = identity
        self._reader = json.loads(open(data_filepath, 'r').read()).items()
        self._parser = ExampleParser(dictionary, self._wave_augmentator, self._spec_augmentator, features_config,
                                     text_processor)

        self._features_extractor = FeatureExtractorFactory.create("example", features_config,
                                                                  identity, identity)
        self._load_batches(sort_by_length)

    def _load_batches(self, sort_by_duration):
        samples = [self._parser.parse(sample) for sample in self._reader]

        if sort_by_duration:
            samples.sort(key=take_first_item)

        self._batches = [SpeechBatch.create_from_list(samples[i:i + self._batch_size], frame_shift=self._frame_shift,
                                                      pad_to=self._pad_to)
                         for i in range(0, len(samples), self._batch_size)]
        if torch.cuda.is_available():
            self._batches = [b.pin_memory() for b in self._batches]

    def _in_memory_stream(self):
        for batch in self._batches:
            yield BatchBlock([batch.clone_to_gpu()], None)

    def __iter__(self, max_sz=None):
        return self._in_memory_stream()

    @property
    def total_batches(self):
        return len(self._batches)

    @property
    def total_samples(self):
        return len(self._reader)


class Kekos:
    def __init__(self,
                 json_table_filepath: str,
                 dictionary: Dictionary,
                 batch_size: int,
                 features_config: dict,
                 text_processor,
                 sort_by_length: bool,
                 pad_to: int,
                 max_duration: int,
                 **kwargs):
        self._json_table_filepath = json_table_filepath
        self._batch_size = batch_size
        self._frame_shift = features_config["frame-shift"]
        self._pad_to = pad_to

        self._wave_augmentator = identity
        self._spec_augmentator = identity

        self._reader = open(json_table_filepath, 'r').readlines()[:1000]
        for i in range(len(self._reader)):
            self._reader[i] = json.loads(self._reader[i].rstrip())
        self._parser = KekosParser(dictionary, self._wave_augmentator, self._spec_augmentator, features_config,
                                   None, None, text_processor)

        self._features_extractor = FeatureExtractorFactory.create("example", features_config,
                                                                  identity, identity)
        self._load_batches(sort_by_length)

    def _load_batches(self, sort_by_duration):
        samples = [self._parser.parse(sample) for sample in self._reader]

        if sort_by_duration:
            samples.sort(key=take_first_item)

        self._batches = [SpeechBatch.create_from_list(samples[i:i + self._batch_size], frame_shift=self._frame_shift,
                                                      pad_to=self._pad_to)
                         for i in range(0, len(samples), self._batch_size)]
        if torch.cuda.is_available():
            self._batches = [b.pin_memory() for b in self._batches]

    def _in_memory_stream(self):
        for batch in self._batches:
            yield BatchBlock([batch.clone_to_gpu()], None)

    def __iter__(self, max_sz=None):
        return self._in_memory_stream()
    
    def __len__(self):
        return len(self._batches)

    @property
    def total_batches(self):
        return len(self._batches)

    @property
    def total_samples(self):
        return len(self._reader)
