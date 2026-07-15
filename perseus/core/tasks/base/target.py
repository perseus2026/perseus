import typing as t
from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl
import torch
from torch import nn

from perseus import utils
from perseus.core import encoders


class Artifacts(ABC):
    @abstractmethod
    def static_transform(self, feature_to_preprocessor: dict[str, encoders.Preprocessor], /) -> t.Self: ...

    @abstractmethod
    def dynamic_transform(self, feature_to_preprocessor: dict[str, encoders.Preprocessor], /) -> t.Any: ...

    @abstractmethod
    def embed(
        self,
        transformed: t.Any,
        feature_to_embedder: dict[str, encoders.Embedder],
        target: t.Any | None = None,
        /,
    ) -> t.Any: ...

    @abstractmethod
    def save(self, path: Path, /) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path, /) -> t.Self: ...


class Preprocessor(ABC):
    @classmethod
    @abstractmethod
    def fit(
        cls,
        values: pl.Series,
        artifacts: Artifacts,
        feature_to_observer: dict[str, encoders.Observer],
        /,
    ) -> t.Self: ...

    @abstractmethod
    def static_transform(self, values: pl.Series, /) -> pl.Series: ...

    @abstractmethod
    def dynamic_transform(self, value: t.Any, /) -> t.Any: ...

    @abstractmethod
    def collate(self, values: list[t.Any], /) -> t.Any: ...

    @abstractmethod
    def save(self, path: Path, /) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path, /) -> t.Self: ...


class Layer(utils.torch.Module, ABC):
    @classmethod
    @abstractmethod
    def init(cls, preprocessor: Preprocessor, dim: int, /) -> t.Self: ...

    @abstractmethod
    def forward(self, backbone_embeddings: torch.Tensor, embedded_artifacts: t.Any, /) -> t.Any: ...


class Criterion(nn.Module, ABC):
    @classmethod
    @abstractmethod
    def init(cls, preprocessor: Preprocessor, layer: Layer, /) -> t.Self: ...

    @abstractmethod
    def forward(
        self,
        backbone_embeddings: torch.Tensor,
        embedded_artifacts: t.Any,
        target: t.Any,
        /,
    ) -> torch.Tensor: ...


class Head(ABC):
    extras: t.ClassVar[list[str]] = []

    @classmethod
    @abstractmethod
    def init(cls, preprocessor: Preprocessor, layer: Layer, /) -> t.Self: ...

    @abstractmethod
    def predict(
        self,
        backbone_embeddings: torch.Tensor,
        embedded_artifacts: t.Any,
        extras: dict[str, t.Any],
        /,
    ) -> pl.Series: ...

    @abstractmethod
    def save(self, path: Path, /) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path, /) -> t.Self: ...
