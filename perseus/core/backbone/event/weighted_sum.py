import torch
from torch import nn

from perseus.core.backbone.event import base


class Aggregator(base.Aggregator):
    def __init__(self, dim: int, num_features: int, /) -> None:
        super().__init__(dim, num_features)

        self.weights = nn.Parameter(torch.ones(num_features, dtype=torch.float32))

    def forward(self, features: torch.Tensor, /) -> torch.Tensor:
        return (features * self.weights.view(-1, 1)).sum(dim=2)
