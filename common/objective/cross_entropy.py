import torch
import torch.nn as nn


class CrossEntropyLoss:
    def __init__(self, pad_id=0, reduction='sum', indices_weight=None, drop_nans=True):
        if indices_weight is not None:
            self._criterion = nn.CrossEntropyLoss(weight=indices_weight, reduction='none')
        else:
            self._criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='none')
        self._reduction = reduction
        self._drop_nans = drop_nans

    def __call__(self, logits, targets):
        pred_flat = logits.reshape(-1, logits.shape[-1])  # (B * T) x vocab_size
        target_flat = targets.view(-1)  # (B * T)
        loss = self._criterion(pred_flat, target_flat)

        if self._drop_nans:
            finite_mask = torch.isfinite(loss)
            finite_ratio = finite_mask.float().mean()
            if finite_ratio > 0.75:
                loss = loss[finite_mask]
                if self._reduction == 'mean':
                    loss *= finite_ratio

        if self._reduction == 'sum':
            loss = torch.sum(loss)
        elif self._reduction == 'mean':
            loss = torch.mean(loss)
        return loss