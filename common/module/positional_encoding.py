import math
import sys

import torch
from torch import Tensor
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float = 0., max_len: int = 5000, concat: bool = False,
                 features_first: bool = False, **kwargs):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._dropout = nn.Dropout(p=dropout)
        self._concat = concat
        self._features_first = features_first
        self._max_len = max_len

        embeddings = torch.zeros(max_len, self._embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self._embedding_dim, 2).float() * (-math.log(10000.0) / self._embedding_dim)
        )
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)
        embeddings.unsqueeze_(0)

        self.register_buffer("_embeddings", embeddings)

    def output_dim(self) -> int:
        return self._embedding_dim

    def forward(self, input: Tensor) -> Tensor:
        if self._features_first:
            input = input.transpose(1, 2)  # (B, F, T) -> (B, T, F)
        if self._concat:
            embeddings_additional = self._embeddings[:, :input.size(1), :].repeat(input.size(0), 1, 1)
            if torch.cuda.is_available():
                embeddings_additional = embeddings_additional.cuda()
            output = torch.cat(
                [input, embeddings_additional],
                dim=-1
            )
        else:
            output = input + self._embeddings[:, :input.size(1), :]
        if self._features_first:
            output = output.transpose(1, 2)
        return self._dropout(output)
