import abc

from asr.models.encoder_decoder import EncoderDecoderModel
from asr.disk_dataset import SpeechBatch
from common.dictionary import Dictionary
from typing import Tuple, List


class EvaluationMethod(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, model: EncoderDecoderModel, dictionary: Dictionary, batch: SpeechBatch) -> \
            Tuple[List[List[str]], List[List[int]], List[List[str]]]:
        pass

    @abc.abstractmethod
    def name(self):
        pass
