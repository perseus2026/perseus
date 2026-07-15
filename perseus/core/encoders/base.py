import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch

from perseus import utils


class Observer(ABC):
    @abstractmethod
    def observe(self, values: pl.Series, /) -> None: ...


class Preprocessor(ABC):
    @classmethod
    @abstractmethod
    def fit(cls, observer: Observer, /, **kwargs: t.Any) -> t.Self: ...

    @abstractmethod
    def static_transform(self, values: pl.Series, /) -> pl.Series: ...

    @abstractmethod
    def dynamic_transform(self, values: pl.Series, /) -> torch.Tensor: ...

    @abstractmethod
    def save(self, path: Path, /) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path, /) -> t.Self: ...


class Embedder(utils.torch.Module, ABC):
    @classmethod
    @abstractmethod
    def init(cls, preprocessor: Preprocessor, dim: int, /, **kwargs: t.Any) -> t.Self: ...

    @abstractmethod
    def forward(self, values: torch.Tensor, /) -> torch.Tensor: ...


@dataclass(frozen=True)
class Encoder:
    observer: type[Observer]
    preprocessor: type[Preprocessor]
    embedder: type[Embedder]
