import torch
from torch import nn

from perseus.core.backbone.event import base


class Aggregator(base.Aggregator):
    def __init__(self, dim: int, num_features: int, /) -> None:
        super().__init__(dim, num_features)

        self.linear = nn.Linear(num_features * dim, dim)

    def forward(self, features: torch.Tensor, /) -> torch.Tensor:
        return self.linear(features.flatten(start_dim=2))
