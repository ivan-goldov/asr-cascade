from evaluation.evaluation_method.generator_evaluation_method import GeneratorEvaluationMethod
from asr.evaluation.generator import GreedyGenerator
from asr.models.encoder_decoder import EncoderDecoderModel
from common.dictionary import Dictionary


class GreedyEvaluation(GeneratorEvaluationMethod):
    def _construct_generator(self, model: EncoderDecoderModel, dictionary: Dictionary) -> GreedyGenerator:
        return GreedyGenerator(model, dictionary)

    def name(self):
        return 'greedy'
