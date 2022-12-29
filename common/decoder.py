from typing import List

from torch import Tensor

from .dictionary import Dictionary


class NNDecoder:
    def decode(self, probs: Tensor) -> Tensor:
        raise NotImplementedError


class GreedyNNDecoder(NNDecoder):
    def decode(self, probs: Tensor) -> Tensor:
        return probs.argmax(dim=-1, keepdim=False).int()


class TextDecoder:
    def __init__(self, dictionary: Dictionary):
        self._dictionary = dictionary

    def decode(self, predictions: Tensor) -> List[str]:
        raise NotImplementedError


class SimpleTextDecoder(TextDecoder):
    def decode(self, predictions: Tensor) -> List[str]:
        hypotheses = []
        prediction_cpu_tensor = predictions.long().cpu()
        # iterate over batch
        for i in range(prediction_cpu_tensor.shape[0]):
            prediction = prediction_cpu_tensor[i].numpy().tolist()
            hypothesis = self._dictionary.decode(prediction).strip()
            hypotheses.append(hypothesis)
        return hypotheses


class CTCTextDecoder(TextDecoder):
    def decode(self, predictions: Tensor) -> List[str]:
        blank_id = self._dictionary.blank_id()
        hypotheses = []
        prediction_cpu_tensor = predictions.long().cpu()
        # iterate over batch
        for i in range(prediction_cpu_tensor.shape[0]):
            prediction = prediction_cpu_tensor[i].numpy().tolist()
            # CTC decoding procedure
            decoded_prediction = []
            previous = blank_id
            for p in prediction:
                if (p != previous or previous == blank_id) and p != blank_id:
                    decoded_prediction.append(p)
                previous = p
            hypothesis = self._dictionary.decode(decoded_prediction)
            hypotheses.append(hypothesis)
        return hypotheses


class Decoder:
    def __init__(self, nn_decoder: NNDecoder, text_decoder: TextDecoder, dictionary: Dictionary):
        self._nn_decoder = nn_decoder
        self._text_decoder = text_decoder
        self._dictionary = dictionary

    def decode(self, probs: Tensor) -> List[str]:
        predictions = self._nn_decoder.decode(probs)
        return self._text_decoder.decode(predictions)

    def decode_probs(self, probs: Tensor) -> Tensor:
        return self._nn_decoder.decode(probs)

    def decode_predictions(self, predictions: Tensor) -> List[str]:
        return self._text_decoder.decode(predictions)

    def dictionary(self) -> Dictionary:
        return self._dictionary


def create_decoder(nn_decoder, text_decoder, dictionary):
    if nn_decoder == "greedy":
        nn_decoder = GreedyNNDecoder()
    else:
        raise ValueError(f"Invalid nn_decoder choice: {nn_decoder}")

    if text_decoder == "simple":
        text_decoder = SimpleTextDecoder(dictionary)
    elif text_decoder == "ctc":
        text_decoder = CTCTextDecoder(dictionary)
    else:
        raise ValueError(f"Invalid text_decoder choice: {text_decoder}")

    return Decoder(nn_decoder, text_decoder, dictionary)
