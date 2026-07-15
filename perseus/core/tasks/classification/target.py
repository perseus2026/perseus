import json
import typing as t
from pathlib import Path

import polars as pl
import torch
from torch import nn

from perseus.core import encoders
from perseus.core.tasks import base


class Artifacts(base.Artifacts):
    def static_transform(self, _feature_to_preprocessor: dict[str, encoders.Preprocessor]) -> t.Self:
        return self

    def dynamic_transform(self, _feature_to_preprocessor: dict[str, encoders.Preprocessor]) -> None:
        return

    def embed(
        self,
        _transformed: t.Any,
        _feature_to_embedder: dict[str, encoders.Embedder],
        _target: t.Any | None = None,
        /,
    ) -> None:
        return

    def save(self, _path: Path, /) -> None:
        return

    @classmethod
    def load(cls, _path: Path, /) -> t.Self:
        return cls()


class Preprocessor(base.Preprocessor):
    def __init__(self, label_to_index: dict[str, int], /) -> None:
        self.label_to_index = label_to_index

    @classmethod
    def fit(
        cls,
        values: pl.Series,
        _artifacts: Artifacts,
        _feature_to_observer: dict[str, encoders.Observer],
        /,
    ) -> t.Self:
        labels = values.unique().sort().to_list()
        label_to_index = {label: index for index, label in enumerate(labels)}
        return cls(label_to_index)

    def static_transform(self, values: pl.Series, /) -> pl.Series:
        return values.replace(self.label_to_index, default=None, return_dtype=pl.UInt32())

    def dynamic_transform(self, value: int) -> int:
        return value

    def collate(self, values: list[int], /) -> torch.LongTensor:
        return torch.tensor(values, dtype=torch.long)

    def save(self, path: Path, /) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "label_to_index.json").write_text(json.dumps(self.label_to_index))

    @classmethod
    def load(cls, path: Path, /) -> t.Self:
        label_to_index = json.loads((path / "label_to_index.json").read_text())
        return cls(label_to_index)


class Layer(base.Layer):
    def __init__(self, dim: int, num_classes: int, /) -> None:
        super().__init__()

        self.linear = nn.Linear(dim, num_classes)

    @classmethod
    def init(cls, preprocessor: Preprocessor, dim: int, /) -> t.Self:
        return cls(dim, len(preprocessor.label_to_index))

    def forward(self, backbone_embeddings: torch.Tensor, _embedded_artifacts: t.Any, /) -> t.Any:
        return self.linear(backbone_embeddings)


class Criterion(base.Criterion):
    def __init__(self, layer: Layer, /) -> None:
        super().__init__()

        self._layer = [layer]

    @classmethod
    def init(cls, _preprocessor: Preprocessor, layer: Layer, /) -> t.Self:
        return cls(layer)

    def forward(
        self,
        backbone_embeddings: torch.Tensor,
        embedded_artifacts: t.Any,
        target: torch.Tensor,
        /,
    ) -> torch.Tensor:
        prediction = self._layer[0](backbone_embeddings, embedded_artifacts)
        return nn.functional.cross_entropy(prediction, target)


class Head(base.Head):
    def __init__(self, layer: Layer, labels: list[str], /) -> None:
        self.layer = layer
        self.labels = labels

    @classmethod
    def init(cls, preprocessor: Preprocessor, layer: Layer, /) -> t.Self:
        return cls(layer, list(preprocessor.label_to_index.keys()))

    def predict(
        self,
        backbone_embeddings: torch.Tensor,
        embedded_artifacts: t.Any,
        _extras: dict[str, t.Any],
        /,
    ) -> pl.Series:
        self.layer.eval().to(backbone_embeddings.device)

        prediction = self.layer(backbone_embeddings, embedded_artifacts).softmax(dim=-1)
        prediction = [dict(zip(self.labels, probas, strict=False)) for probas in prediction.tolist()]
        return pl.Series("prediction", prediction, pl.Struct({label: pl.Float32() for label in self.labels}))

    def save(self, path: Path, /) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.layer.save(path / "layer")
        (path / "labels.json").write_text(json.dumps(self.labels))

    @classmethod
    def load(cls, path: Path, /) -> t.Self:
        layer = Layer.load(path / "layer")
        labels = json.loads((path / "labels.json").read_text())
        return cls(layer, labels)
