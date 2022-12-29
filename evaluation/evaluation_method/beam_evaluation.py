from evaluation.evaluation_method.generator_evaluation_method import GeneratorEvaluationMethod
from asr.evaluation.generator import BeamGenerator
from asr.models.encoder_decoder import EncoderDecoderModel
from common.dictionary import Dictionary
from typing import Callable


def construct_references(decoder, tokens, tokens_lengths):
    targets_cpu = tokens.long().cpu()
    lengths_cpu = tokens_lengths.long().cpu()
    references = []
    for i in range(targets_cpu.size(0)):
        length = lengths_cpu[i].item()
        target = targets_cpu[i][:length].numpy().tolist()
        reference = decoder.dictionary().decode(target)
        references.append(reference)
    return references


class BeamEvaluation(GeneratorEvaluationMethod):
    def __init__(self, beam_size: int, norm: Callable[[int], float]):
        self._beam_size = beam_size
        self._norm = norm

    def _construct_generator(self, model: EncoderDecoderModel, dictionary: Dictionary):
        return BeamGenerator(model, dictionary, beam_size=self._beam_size, norm=self._norm)

    def name(self):
        return f'beam(size={self._beam_size}, norm={self._norm})'
