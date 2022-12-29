import torch.nn as nn


class Linear(nn.Linear):
    def input_dim(self) -> int:
        return self.in_features

    def output_dim(self) -> int:
        return self.out_features
