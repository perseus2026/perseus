from abc import ABC, abstractmethod

import torch
from torch import nn


class Aggregator(nn.Module, ABC):
    @abstractmethod
    def __init__(self, dim: int, max_events_per_sequence: int, /) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self,
        embeddings: torch.Tensor,
        positions: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor,
        /,
    ) -> torch.Tensor: ...
