import heapq
from typing import List, Tuple, Callable

import numpy as np

import torch
import torch.nn.functional as F

from asr.models import EncoderDecoderModel, EncoderResult, DecoderResult
from common.dictionary import Dictionary


class Generator:
    def __call__(self, sample: torch.Tensor) -> List[int]:
        raise NotImplementedError


class GreedyGenerator(Generator):
    def __init__(self, model: EncoderDecoderModel, dictionary: Dictionary):
        self._model = model
        self._bos_id = dictionary.bos_id()
        self._eos_id = dictionary.eos_id()

    def __call__(self, sample: torch.Tensor) -> List[List[int]]:
        max_n_steps = sample.size(2)
        device = sample.device

        encoder_result: EncoderResult = self._model.encoder(sample, None)

        speakers_num = self._model.max_speakers_num()
        prev_tokens = [[self._bos_id] for _ in range(speakers_num)]
        lens = [-1 for _ in range(speakers_num)]
        for _ in range(max_n_steps):
            prev_tokens_tensor = [torch.LongTensor(t).unsqueeze(0).to(device) for t in prev_tokens]
            decoder_result: DecoderResult = self._model.decoder(encoder_result=encoder_result,
                                                                texts=prev_tokens_tensor,
                                                                text_lengths=[None for _ in range(speakers_num)])
            next_tokens = [decoder_result.output[s][0, -1].argmax().item() for s in range(speakers_num)]
            for s in range(speakers_num):
                prev_tokens[s].append(next_tokens[s])
            for s in range(speakers_num):
                if next_tokens[s] == self._eos_id:
                    lens[s] = len(prev_tokens[s])
            if all(l != -1 for l in lens):
                break

        return [prev_tokens[s][:lens[s]] for s in range(speakers_num)]


ALPHA = 0.1


def alpha_norm(step):
    return ((5 + step) / 6) ** ALPHA


class PowNorm:
    def __init__(self, pow: float):
        self._pow = pow

    def __call__(self, step: int) -> float:
        return step ** self._pow

    def __str__(self):
        return f'pow_norm({self._pow})'


sqrt_norm = PowNorm(0.5)
none = PowNorm(1)
sqr_norm = PowNorm(2)


class BeamGenerator(Generator):
    def __init__(self, model: EncoderDecoderModel, dictionary: Dictionary, beam_size: int,
                 norm: Callable[[int], float] = sqr_norm):
        self._model = model
        self._bos_id = dictionary.bos_id()
        self._eos_id = dictionary.eos_id()
        self._beam_size = beam_size

        self.norm = norm

    def __perform_beam_search(self, max_n_steps, encoder_result, model_decoder, device):
        speakers_num = self._model.max_speakers_num()
        partial_hypotheses = [(0, [[self._bos_id] for _ in range(speakers_num)])]
        final_hypotheses = []
        while len(partial_hypotheses):
            curr_partial_score, curr_partial_hypotheses = heapq.heappop(partial_hypotheses)

            # if there is a final hypothesis with a better score then stop
            if final_hypotheses and final_hypotheses[0][0] < curr_partial_score:
                break

            prev_tokens_tensor = [torch.LongTensor(curr_partial_hyp).unsqueeze(0).to(device)
                                  for curr_partial_hyp in curr_partial_hypotheses]
            decoder_result: DecoderResult = model_decoder(encoder_result=encoder_result,
                                                          texts=prev_tokens_tensor,
                                                          text_lengths=[None for _ in range(speakers_num)])
            next_tokens_logits = [decoder_result.output[s][0, -1] for s in range(speakers_num)]
            bs = self._beam_size

            next_tokens_logprobs = [F.log_softmax(nl, dim=0) for nl in next_tokens_logits]
            topk_logprobs = [next_tokens_logprobs[s].topk(bs) for s in range(speakers_num)]

            topk_continuations = torch.zeros(tuple(bs for _ in range(speakers_num))).to(device)
            for s in range(speakers_num):
                topk_continuations += torch.unsqueeze(topk_logprobs[s].values, speakers_num - s - 1) \
                    .expand(topk_continuations.shape)
            topk_continuations = topk_continuations.reshape(-1).topk(bs)

            for token_score, token_id in zip(topk_continuations.values, topk_continuations.indices):
                token_score = float(token_score)
                token_ids = []
                for s in range(speakers_num):
                    token_ids.append(int(topk_logprobs[s].indices[int(token_id // bs)]))
                    token_id = token_id % bs

                l = len(curr_partial_hypotheses[0])
                old_denorm_score = curr_partial_score * self.norm(l)
                new_score = (old_denorm_score - token_score) / self.norm(l + 1)

                new_hypotheses = [curr_partial_hypotheses[s] + [token_ids[s]] for s in range(speakers_num)]
                new_item = (new_score, new_hypotheses)

                if token_id == self._eos_id or len(new_hypotheses[0]) >= max_n_steps:
                    heapq.heappush(final_hypotheses, new_item)
                else:
                    heapq.heappush(partial_hypotheses, new_item)

            if len(partial_hypotheses) > self._beam_size:
                partial_hypotheses = heapq.nsmallest(self._beam_size, partial_hypotheses)
                heapq.heapify(partial_hypotheses)

        return final_hypotheses[0][1]

    def __call__(self, sample: torch.Tensor) -> List[int]:
        max_n_steps = sample.size(2)
        device = sample.device

        encoder_result: EncoderResult = self._model.encoder(sample, None)

        predictions = self.__perform_beam_search(max_n_steps, encoder_result, self._model.decoder, device)

        return predictions
