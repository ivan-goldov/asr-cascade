import copy
from typing import Optional

import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, ModuleList, MultiheadAttention, Dropout, Linear, LayerNorm

from common.dictionary import Dictionary


class TransformerEncoder(Module):
    def __init__(self, model_definition: dict, **kwargs):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model=model_definition["d_model"],
                                                num_heads=model_definition["num_heads"],
                                                dim_feedforward=model_definition["dim_feedforward"],
                                                dropout=model_definition["dropout"],
                                                activation=model_definition["activation"],
                                                normalize_before=model_definition["normalize_before"])
        """encoder_layer = TransformerEncoderLayer(d_model=model_definition["d_model"],
                                                nhead=model_definition["num_heads"],
                                                dim_feedforward=model_definition["dim_feedforward"],
                                                dropout=model_definition["dropout"],
                                                activation=model_definition["activation"])"""
        self.num_layers = model_definition["num_layers"]
        self.d_model = model_definition["d_model"]
        self.layers = _get_clones(encoder_layer, self.num_layers)
        self.final_norm = LayerNorm(self.d_model) if model_definition["final_norm"] else None

    def input_dim(self) -> int:
        return self.d_model

    def output_dim(self) -> int:
        return self.d_model

    def forward(self, input: Tensor, square_mask: Optional[Tensor] = None,
                padding_mask: Optional[Tensor] = None, final_norm: bool = True) -> Tensor:
        output = input.transpose(1, 2).transpose(0, 1)  # (B, F, T) -> (B, T, F) -> (T, B, F)

        for layer in self.layers:
            #print(f'device1 = {output.device}, device2 = {padding_mask.device}')
            output = layer(output, src_mask=square_mask, src_key_padding_mask=padding_mask)

        if self.final_norm is not None and final_norm:
            output = self.final_norm(output)

        output.transpose_(0, 1).transpose_(1, 2)  # (T, B, F) -> (B, T, F) -> (B, F, T)

        return output


class TransformerDecoder(Module):
    def __init__(self, model_definition: dict, dictionary: Dictionary, **kwargs):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(d_model=model_definition["d_model"],
                                                num_heads=model_definition["num_heads"],
                                                dim_feedforward=model_definition["dim_feedforward"],
                                                dropout=model_definition["dropout"],
                                                activation=model_definition["activation"],
                                                normalize_before=model_definition["normalize_before"])
        self.num_layers = model_definition["num_layers"]
        self.d_model = model_definition["d_model"]
        self.layers = _get_clones(decoder_layer, self.num_layers)
        self.final_norm = LayerNorm(self.d_model) if model_definition["final_norm"] else None
        self.final_linear = Linear(self.d_model, 2 * len(dictionary))  # KEK

    def input_dim(self) -> int:
        return self.d_model

    def output_dim(self) -> int:
        return self.final_linear.out_features

    def forward(self, input: Tensor, memory: Tensor, input_square_mask: Optional[Tensor] = None,
                memory_square_mask: Optional[Tensor] = None, input_padding_mask: Optional[Tensor] = None,
                memory_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = input.transpose(0, 1)  # (B, T, F) -> (T, B, F)
        memory = memory.transpose(1, 2).transpose(0, 1)

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=input_square_mask,
                           memory_mask=memory_square_mask,
                           tgt_key_padding_mask=input_padding_mask,
                           memory_key_padding_mask=memory_padding_mask)

        if self.final_norm is not None:
            output = self.final_norm(output)

        output = output.transpose(0, 1)  # (T, B, F) -> (B, T, F)
        output = self.final_linear(output)  # (B, T, F) -> (B, T, vocab_size)

        return output


class TransformerEncoderLayer(Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1, activation="relu", normalize_before=True):
        super().__init__()
        self.d_model = d_model

        self.self_attn = MultiheadAttention(d_model, num_heads, dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        residual = src
        
        if self.normalize_before:
            src = self.norm1(src)
        
        src = src.float()
        
        #torch.save(self.self_attn.state_dict(), '/home/jupyter/work/resources/self_attn_weight.pt')
        
        #print(self.self_attn.state_dict()['in_proj_weight'].dtype, self.self_attn.state_dict()['in_proj_bias'].dtype)
        #print(src.dtype)
        
        #torch.save(src, '/home/jupyter/work/resources/source1.pt')
        #if src_mask is not None:
        #    torch.save(src_mask, '/home/jupyter/work/resources/src_mask.pt')
        #if src_key_padding_mask is not None: 
        #    torch.save(src_key_padding_mask, '/home/jupyter/work/resources/src_key_padding_mask.pt')
        
        src = self.self_attn(src, src, src, attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.dropout2(src)
        src = residual + src
        if not self.normalize_before:
            src = self.norm2(src)

        return src


class TransformerDecoderLayer(Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1, activation="relu", normalize_before=True):
        super().__init__()
        self.d_model = d_model

        self.self_attn = MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, num_heads, dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        
        tgt = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                             key_padding_mask=tgt_key_padding_mask)[0]
        tgt = residual + self.dropout1(tgt)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        tgt = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                  key_padding_mask=memory_key_padding_mask)
        tgt = tgt[0]
        tgt = residual + self.dropout2(tgt)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm3(tgt)
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.dropout3(tgt)
        tgt = residual + tgt
        if not self.normalize_before:
            tgt = self.norm3(tgt)

        return tgt


def _get_clones(module, n):
    return ModuleList([copy.deepcopy(module) for _ in range(n)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")
