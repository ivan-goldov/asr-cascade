import torch.nn as nn


class CTCLossNM:
    def __init__(self, **kwargs):
        self._blank = kwargs["blank_id"]
        self._criterion = nn.CTCLoss(blank=self._blank, reduction="sum")

    def __call__(self, log_probs, targets, input_length, target_length):
        input_length = input_length.long()
        target_length = target_length.long()
        targets = targets.long()
        loss = self._criterion(log_probs.transpose(1, 0),
                               targets,
                               input_length,
                               target_length)
        return loss
