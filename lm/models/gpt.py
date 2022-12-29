from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn

from common.dictionary import Dictionary
from common.utils import get_attention_mask, get_padding_mask
from common.module import Embedding, PositionalEncoding, TransformerEncoder


class GPT(nn.Module):
    def __init__(self, model_config: dict, dictionary: Dictionary):
        super().__init__()

        self.embedding = Embedding(dictionary=dictionary, **model_config["embedding"])
        self.positional_encoding = PositionalEncoding(**model_config["positional_encoding"])
        self.dropout = nn.Dropout(model_config["transformer"]["dropout"])
        self.transformer_encoder = TransformerEncoder(model_config["transformer"])
        self.final_linear = nn.Linear(self.transformer_encoder.d_model, len(dictionary), bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.final_linear.weight, mean=0, std=self.transformer_encoder.d_model ** -0.5)

    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        return self

    def forward(self, input: Tensor, lengths: Optional[Tensor] = None, final_norm: bool = True,
                features_only: bool = False):
        output = self.embedding(input)
        output = self.positional_encoding(output)
        output = self.dropout(output)

        output.transpose_(1, 2)  # (B, T, F) -> (B, F, T)
        padding_mask = get_padding_mask(lengths, output.size(-1)) if lengths is not None else None
        output = self.transformer_encoder(input=output,
                                          square_mask=get_attention_mask(output.size(-1)).to(output.device),
                                          padding_mask=padding_mask,
                                          final_norm=final_norm)
        output.transpose_(1, 2)  # (B, F, T) -> (B, T, F)

        if not features_only:
            output = self.final_linear(output)

        return output


def load_from_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    last_training_params = checkpoint["progress_tracker_state"]["params_history"][-1]
    model_definition = last_training_params["model_definition"]
    dictionary = last_training_params["dictionary"]
    model = GPT(model_definition, dictionary)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, model_definition
