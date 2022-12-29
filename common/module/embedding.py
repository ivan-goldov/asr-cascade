import math

import torch.nn as nn
from torch import Tensor
from typing import Optional

from common.dictionary import Dictionary


class Embedding(nn.Module):
    def __init__(self, embedding_dim: int, dictionary: Dictionary, scale_embedding: bool = False,
                 max_norm: Optional[float] = None, norm_type: float = 2., **kwargs):
        super().__init__()
        vocab_size = len(dictionary)
        self._pad_id = dictionary.pad_id()
        self._embedding_dim = embedding_dim
        self._embedding = nn.Embedding(num_embeddings=vocab_size,
                                       embedding_dim=embedding_dim,
                                       padding_idx=self._pad_id,
                                       max_norm=max_norm,
                                       norm_type=norm_type)
        self._scale = math.sqrt(self._embedding_dim) if scale_embedding else 1
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self._embedding.weight, mean=0, std=self._embedding_dim ** -0.5)
        nn.init.constant_(self._embedding.weight[self._pad_id], 0)

    def output_dim(self) -> int:
        return self._embedding_dim

    def forward(self, input: Tensor) -> Tensor:
        return self._embedding(input) * self._scale
