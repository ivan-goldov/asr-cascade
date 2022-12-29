from typing import List, Optional

import torch
from torch import Tensor
import torch.nn as nn

from asr.disk_dataset import SpeechBatch
from common.dictionary import Dictionary
from common.module import Coalescence, Embedding, PositionalEncoding, Linear, TransformerEncoder, TransformerDecoder, \
    JasperEncoder, JasperDecoderForCTC
from common.utils import get_attention_mask, get_padding_mask
from lm.models import GPT


class ModelResult:
    def __init__(self, output: List[Tensor], encoded_lengths: Tensor, model_context: dict):
        self._output = output
        self.max_speakers_num = len(output)
        self._encoded_lengths = encoded_lengths
        self._model_context = model_context

        self._probs = [None for _ in range(self.max_speakers_num)]
        self._log_probs = [None for _ in range(self.max_speakers_num)]

    @property
    def output(self) -> List[Tensor]:
        return [self._output[s] for s in range(self.max_speakers_num)]

    @property
    def encoded_lengths(self) -> Tensor:
        return self._encoded_lengths

    @property
    def model_context(self) -> dict:
        return self._model_context

    def _speaker_probs(self, s: int) -> Tensor:
        if self._probs[s] is None:
            self._probs[s] = nn.functional.softmax(self._output[s], dim=-1)
        return self._probs[s]

    @property
    def probs(self) -> List[Tensor]:
        return [self._speaker_probs(s) for s in range(self.max_speakers_num)]

    def _speaker_log_probs(self, s: int) -> Tensor:
        if self._log_probs[s] is None:
            self._log_probs[s] = nn.functional.log_softmax(self._output[s], dim=-1)
        return self._log_probs[s]

    @property
    def log_probs(self) -> List[Tensor]:
        return [self._speaker_log_probs(s) for s in range(self.max_speakers_num)]


class EncoderDecoderModel(nn.Module):
    def __init__(self, model_config: dict, input_dim: int, dictionary: Dictionary, max_speakers_num: int,
                 language_model: Optional[GPT] = None):
        super().__init__()

        self._model_context = {"transformer_decoder": False, "freeze_lm": False}

        self.encoder = Encoder(model_config["encoder"], input_dim, dictionary, self._model_context)
        self.decoder = Decoder(model_config["decoder"], self.encoder.output_dim(), dictionary, self._model_context,
                               max_speakers_num=max_speakers_num, language_model=language_model)
        self._max_speakers_num = max_speakers_num

        if self._model_context["transformer_decoder"]:
            self._pad_id = dictionary.pad_id()
            self._bos_id = dictionary.bos_id()
            self._eos_id = dictionary.eos_id()

    def max_speakers_num(self) -> int:
        return self._max_speakers_num

    def num_weights(self) -> int:
        return self.encoder.num_weights() + self.decoder.num_weights()

    def train(self, mode=True):
        super().train(mode)
        if self._model_context["freeze_lm"]:
            self.decoder._language_model.eval()
        return self

    def forward(self, batch: SpeechBatch) -> ModelResult:
        tokens, tokens_lengths = batch.tokens.copy(), batch.tokens_lengths.copy()
        if self._model_context["transformer_decoder"]:
            for s in range(self._max_speakers_num):
                if tokens[s] is not None:
                    tokens[s] = self._insert_bos_id(tokens[s])
                    tokens_lengths[s] = tokens_lengths[s] + 1
        
#         LOG.info('Will add bos id')
#         LOG.info(tokens[0])
#         LOG.info(tokens_lengths[0])
#         LOG.info('')
#         LOG.info(tokens[0])
#         LOG.info(tokens_lengths[0])
#         raise Exception('kek')

        encoder_result: EncoderResult = self.encoder(batch.features, batch.features_lengths)
        decoder_result: DecoderResult = self.decoder(encoder_result, tokens, tokens_lengths)

        return ModelResult(output=decoder_result.output,
                           encoded_lengths=encoder_result.encoded_lengths,
                           model_context=self._model_context)

    def _insert_bos_id(self, tokens: Tensor) -> Tensor:
        batch_size = tokens.size(0)
        x = torch.zeros(batch_size, 1).fill_(self._bos_id).long().to(tokens.device)
        return torch.cat((x, tokens), dim=1)


class EncoderResult:
    def __init__(self, output: Tensor, encoded_lengths: Tensor, transformer_encoder_padding_mask: Tensor):
        self._output = output
        self._encoded_lengths = encoded_lengths
        self._transformer_encoder_padding_mask = transformer_encoder_padding_mask

    @property
    def output(self) -> Tensor:
        return self._output

    @property
    def encoded_lengths(self) -> Tensor:
        return self._encoded_lengths

    @property
    def transformer_encoder_padding_mask(self) -> Tensor:
        return self._transformer_encoder_padding_mask


class Encoder(nn.Module):
    def __init__(self, encoder_config: List[dict], input_dim: int, dictionary: Dictionary, model_context: dict):
        super().__init__()

        self._input_dim = input_dim
        self._layers = nn.ModuleList()

        for i in range(len(encoder_config)):
            layer_config = encoder_config[i].copy()
            layer_name = layer_config.pop("name")
            layer_config["dictionary"] = dictionary
            self._layers.extend(_create_layer(layer_name, layer_config, input_dim))
            input_dim = self._layers[-1].output_dim()

        self._output_dim = self._layers[-1].output_dim()

    def input_dim(self) -> int:
        return self._input_dim

    def output_dim(self) -> int:
        return self._output_dim

    def num_weights(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input: Tensor, lengths: Tensor) -> EncoderResult:
        output = input
        transformer_encoder_padding_mask = None
        for layer in self._layers:
            if isinstance(layer, (Linear, PositionalEncoding)):
                output = layer(output)
            elif isinstance(layer, (Coalescence, JasperEncoder)):
                output, lengths = layer(output, lengths)
            elif isinstance(layer, TransformerEncoder):
                if lengths is not None:
                    transformer_encoder_padding_mask = get_padding_mask(lengths, output.size(-1))
                else:
                    transformer_encoder_padding_mask = None
                output = layer(output, padding_mask=transformer_encoder_padding_mask)
            else:
                raise Exception(f"Unknown encoder layer: {layer}")

        return EncoderResult(output, lengths, transformer_encoder_padding_mask)


class DecoderResult:
    def __init__(self, output: List[Tensor]):
        self._output = output

    @property
    def output(self) -> List[Tensor]:
        return self._output


class Decoder(nn.Module):
    def __init__(self, decoder_config: List[dict], input_dim: int, dictionary: Dictionary, model_context: dict,
                 max_speakers_num: int, language_model: Optional[GPT] = None):
        super().__init__()

        self._language_model = language_model
        if language_model is not None:
            assert decoder_config[0]["name"] == "language_model"
            if decoder_config[0]["freeze"]:
                self._language_model.freeze()
                model_context["freeze_lm"] = True

        self._layers = nn.ModuleList()
        self._text_layers = nn.ModuleList()

        tokens_dim = None
        self._final_text_layer = Linear(max_speakers_num * input_dim, input_dim)
        for i in range(len(decoder_config)):
            if decoder_config[i]["name"] == "language_model":
                continue
            layer_config = decoder_config[i].copy()
            layer_name = layer_config.pop("name")
            layer_config["dictionary"] = dictionary
            if layer_name in ("embedding", "positional_encoding"):
                self._text_layers.extend(_create_layer(layer_name, layer_config, tokens_dim))
                tokens_dim = self._text_layers[-1].output_dim()
            else:
                self._layers.extend(_create_layer(layer_name, layer_config, input_dim))
                input_dim = self._layers[-1].output_dim()

            if layer_name == "transformer_decoder":
                model_context["transformer_decoder"] = True

    def num_weights(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, encoder_result: EncoderResult, texts: List[Tensor], text_lengths: List[Tensor]) -> DecoderResult:
        speakers_num = len(texts)
        texts_max_len = [text.shape[1] for text in texts]
        max_shape = max(texts_max_len)
        texts = texts.copy()
        for s in range(speakers_num):
            if max_shape == texts_max_len[s]:
                continue
            texts[s] = torch.cat((texts[s],
                                  torch.zeros((texts[s].shape[0],
                                               max_shape - texts_max_len[s]), dtype=torch.long).to(texts[s].device)), 1)

        if self._language_model is not None:
            texts = [self._language_model(t, t_lens, final_norm=True, features_only=True)
                              for t, t_lens in zip(texts, text_lengths)]
        else:
            for layer in self._text_layers:
                texts = [layer(text) for text in texts]

        texts = torch.cat(tuple(texts), 2)
        text_lengths = torch.max(torch.stack(tuple(text_lengths)), 0).values \
            if all(len is not None for len in text_lengths) else None
        texts = self._final_text_layer(texts)

        output = encoder_result.output
        for layer in self._layers:
            if isinstance(layer, (Linear, JasperDecoderForCTC)):
                output = layer(output)
            elif isinstance(layer, TransformerDecoder):
                if text_lengths is not None:
                    tgt_padding_mask = get_padding_mask(text_lengths)
                else:
                    tgt_padding_mask = None
                tgt_mask = get_attention_mask(texts.size(1)).to(texts.device)
                output = layer(input=texts,
                               memory=output,
                               input_square_mask=tgt_mask,
                               input_padding_mask=tgt_padding_mask,
                               memory_padding_mask=encoder_result.transformer_encoder_padding_mask)
            else:
                raise Exception(f"Unknown decoder layer: {layer}")

        outputs = []
        l = output.shape[2] // speakers_num
        for s in range(speakers_num):
            s_output = output[:, :, (l * s):(l * s + l)]
            outputs.append(s_output)
        return DecoderResult(outputs)


class InferenceModel(nn.Module):
    def __init__(self, model: EncoderDecoderModel):
        super().__init__()

        self._model = model

    def num_weights(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def feature_count(self) -> int:
        return self._model.encoder.input_dim()

    def forward(self, features: Tensor, prev_output_tokens: Optional[Tensor] = None) -> Tensor:
        encoder_result: EncoderResult = self._model.encoder(features, None)
        decoder_result: DecoderResult = self._model.decoder(encoder_result, prev_output_tokens, None)
        return decoder_result.output


def _create_layer(name: str, config: dict, input_dim: int) -> List[nn.Module]:
    layer_types = {
        "embedding": Embedding,
        "positional_encoding": PositionalEncoding,
        "coalescence": Coalescence,
        "jasper_encoder": JasperEncoder,
        "jasper_decoder": JasperDecoderForCTC,
        "transformer_encoder": TransformerEncoder,
        "transformer_decoder": TransformerDecoder
    }

    layers = []

    expected_input_dim = config.get("input_dim")
    # if the layer takes an arbitrary dimension, specify the current input dimension in case the layer needs it
    if expected_input_dim is None:
        config["input_dim"] = input_dim
    # if the layer works with a specific dimension, add a linear layer to match the dimensions
    elif expected_input_dim is not None and input_dim != expected_input_dim:
        layers.append(Linear(input_dim, expected_input_dim))

    layers.append(layer_types[name](**config))

    return layers
