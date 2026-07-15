import json
import typing as t
from pathlib import Path

import polars as pl
import torch
from torch import nn

from perseus.core.encoders import base


class Observer(base.Observer):
    def __init__(self) -> None:
        self.values = pl.Series("value", dtype=pl.Unknown())

    def observe(self, values: pl.Series, /) -> None:
        if self.values.is_empty():
            self.values = self.values.cast(values.dtype)
        self.values = pl.concat([self.values, values]).drop_nulls()


class Preprocessor(base.Preprocessor):
    def __init__(self, thresholds: list[float], /) -> None:
        self.thresholds = torch.tensor(thresholds, dtype=torch.float32)

    @property
    def num_bins(self) -> int:
        return len(self.thresholds) - 1

    @classmethod
    def fit(cls, observer: Observer, /, *, max_bins: int = 256) -> t.Self:
        values = observer.values.cast(pl.Float32()).drop_nulls().sort()
        thresholds = []
        for bin_ in range(max_bins + 1):
            quantile = bin_ / max_bins
            position = quantile * (len(values) - 1)
            index = int(position)
            weight = position - index
            threshold = values[index] if weight == 0 else values[index] + weight * (values[index + 1] - values[index])
            thresholds.append(threshold)
        return cls(sorted(set(thresholds)))

    def static_transform(self, values: pl.Series, /) -> pl.Series:
        return values.clone()

    def dynamic_transform(self, values: pl.Series, /) -> torch.Tensor:
        return (
            (
                (values.cast(pl.Float32()).to_torch().unsqueeze(dim=-1) - self.thresholds[:-1])
                / (self.thresholds[1:] - self.thresholds[:-1])
            )
            .clamp(0.0, 1.0)
            .nan_to_num(0.0)
        )

    def save(self, path: Path, /) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "thresholds.json").write_text(json.dumps(self.thresholds.tolist()))

    @classmethod
    def load(cls, path: Path, /) -> t.Self:
        return cls(json.loads((path / "thresholds.json").read_text()))


class Embedder(base.Embedder):
    def __init__(self, num_bins: int, dim: int, /) -> None:
        super().__init__()

        self.linear = nn.Linear(num_bins, dim)

    @classmethod
    def init(cls, preprocessor: Preprocessor, dim: int, /) -> t.Self:
        return cls(preprocessor.num_bins, dim)

    def forward(self, values: torch.Tensor, /) -> torch.Tensor:
        return self.linear(values)


encoder = base.Encoder(Observer, Preprocessor, Embedder)
