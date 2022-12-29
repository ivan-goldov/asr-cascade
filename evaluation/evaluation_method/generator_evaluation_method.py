import abc
import torch

from evaluation.evaluation_method.evaluation_method import EvaluationMethod
from asr.disk_dataset import SpeechBatch
from asr.models.encoder_decoder import EncoderDecoderModel
from common.decoder import create_decoder
from common.dictionary import Dictionary
from typing import Tuple, List


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


class GeneratorEvaluationMethod(EvaluationMethod):
    def evaluate(self, model: EncoderDecoderModel, dictionary: Dictionary, batch: SpeechBatch) -> \
            Tuple[List[List[str]], List[List[int]], List[List[str]]]:
        if torch.cuda.is_available():
            batch = batch.cuda()
        hypotheses = []
        predictions = []
        generator = self._construct_generator(model, dictionary)
        decoder = create_decoder(nn_decoder="greedy", text_decoder="simple", dictionary=dictionary)
        for features, features_len in zip(batch.features, batch.features_lengths):
            f = features[:, :features_len]
            preds = generator(f.reshape(1, f.shape[0], f.shape[1]))
            h, p = [], []
            for s, pred in enumerate(preds):
                h.append(decoder.dictionary().decode(pred))
                p.append(pred)
            hypotheses.append(h)
            predictions.append(p)
        references = [construct_references(decoder, t, t_len)
                      for t, t_len in zip(batch.tokens, batch.tokens_lengths)]
        references = [[references[j][i] for j in range(len(references))] for i in range(len(references[0]))]
        return hypotheses, predictions, references

    @abc.abstractmethod
    def _construct_generator(self, model: EncoderDecoderModel, dictionary: Dictionary):
        pass
