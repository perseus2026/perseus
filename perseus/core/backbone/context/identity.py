import torch

from perseus.core.backbone.context import base


class Aggregator(base.Aggregator):
    def __init__(self, dim: int, num_features: int, /) -> None:
        super().__init__(dim, num_features)

    def forward(self, features: torch.Tensor, /) -> torch.Tensor:
        return features
