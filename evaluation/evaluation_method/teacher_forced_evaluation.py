import torch

from asr.disk_dataset import SpeechBatch
from asr.helpers import eval_quality
from asr.models.encoder_decoder import EncoderDecoderModel
from asr.train import ModelResult
from common.decoder import create_decoder
from common.dictionary import Dictionary
from evaluation.evaluation_method.evaluation_method import EvaluationMethod
from typing import Tuple, List


class TeacherForcedEvaluation(EvaluationMethod):
    def evaluate(self, model: EncoderDecoderModel, dictionary: Dictionary, batch: SpeechBatch) -> \
            Tuple[List[List[str]], List[List[int]], List[List[str]]]:
        with torch.cuda.amp.autocast() and torch.no_grad():
            if torch.cuda.is_available():
                batch = batch.cuda()
            result: ModelResult = model(batch)
        decoder = create_decoder(nn_decoder="greedy", text_decoder="simple", dictionary=dictionary)
        predictions = [decoder.decode_probs(log_probs) for log_probs in result.log_probs]
        _, hypotheses, references = eval_quality(predictions, batch.tokens, batch.tokens_lengths, decoder)

        hypotheses = [[hypotheses[j][i] for j in range(len(hypotheses))] for i in range(len(hypotheses[0]))]
        references = [[references[j][i] for j in range(len(references))] for i in range(len(references[0]))]
        return hypotheses, [list(p) for p in predictions], references

    def name(self):
        return 'teacher_forced'
