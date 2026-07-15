import json
import typing as t
from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl
import torch
from torch import nn

from perseus import utils
from perseus.core import encoders
from perseus.core.tasks import base


class Artifacts(base.Artifacts):
    def __init__(self, items: pl.DataFrame) -> None:
        if "_item" not in items.schema:
            items = items.with_columns(_item="item")  # item also can be feature, so preserve original item
        self.items = items

    def static_transform(self, feature_to_preprocessor: dict[str, encoders.Preprocessor], /) -> t.Self:
        self.items = self.items.with_columns(
            pl.col(feature).map_batches(preprocessor.static_transform)
            for feature, preprocessor in feature_to_preprocessor.items()
        )
        return self

    def dynamic_transform(
        self,
        feature_to_preprocessor: dict[str, encoders.Preprocessor],
        /,
    ) -> tuple[list[t.Any], dict[str, torch.Tensor]]:
        items = self.items["_item"].to_list()
        features = {
            feature: preprocessor.dynamic_transform(self.items[feature])
            for feature, preprocessor in feature_to_preprocessor.items()
        }
        return items, features

    def embed(
        self,
        transformed: tuple[list[t.Any], dict[str, torch.Tensor]],
        feature_to_embedder: dict[str, encoders.Embedder],
        target: t.Any | None = None,
        /,
    ) -> tuple[list[t.Any], torch.Tensor]:
        items, features = transformed
        if target is not None:
            _, item_indices, _ = target
            item_indices = item_indices.tolist()
            items = [items[index] for index in item_indices]
            features = {feature: values[item_indices] for feature, values in features.items()}
        features = torch.stack(
            [
                embedder(features[feature].to(utils.torch.device, non_blocking=True))
                for feature, embedder in feature_to_embedder.items()
            ],
            dim=1,
        )
        return items, features

    def save(self, path: Path, /) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.items.write_parquet(path / "items.pq")

    @classmethod
    def load(cls, path: Path, /) -> t.Self:
        items = pl.read_parquet(path / "items.pq")
        return cls(items)


class Preprocessor(base.Preprocessor):
    def __init__(self, labels: list[str], items: pl.DataFrame, num_features: int, /) -> None:
        super().__init__()

        self.labels = labels
        self.items = items
        self.num_features = num_features

    @classmethod
    def fit(
        cls,
        values: pl.Series,
        artifacts: Artifacts,
        feature_to_observer: dict[str, encoders.Observer],
        /,
    ) -> t.Self:
        labels = [field.name for field in values.dtype.inner.to_schema()["labels"].fields]
        items = (
            values.to_frame("target")
            .lazy()
            .select(items=pl.col("target").list.eval(pl.element().struct.field("item")))
            .explode("items")
            .rename({"items": "item"})
            .unique()
            .sort("item")
            .with_row_index()
            .collect()
        )
        artifacts.items = artifacts.items.join(items.select("item"), how="inner", on="item").sort("item")
        for feature, observer in feature_to_observer.items():
            observer.observe(artifacts.items[feature])
        return cls(labels, items, len(feature_to_observer))

    def static_transform(self, values: pl.Series, /) -> pl.Series:
        return (
            values.to_frame("items")
            .lazy()
            .with_row_index("row_index")
            .explode("items")
            .unnest("items")
            .unnest("labels")
            .join(self.items.lazy().select("item", item_index="index"), how="left", on="item", maintain_order="left")
            .group_by("row_index", maintain_order=True)
            .agg(
                labeled_item_indices=pl.struct(
                    item_indices=pl.col("item_index").implode(),
                    labels=pl.concat_arr(*self.labels).implode(),
                ),
            )
            .select("labeled_item_indices")
            .collect()
            .to_series()
        )

    def dynamic_transform(self, value: dict[str, t.Any], /) -> tuple[list[int], list[list[int]]]:
        return value["item_indices"], value["labels"]

    def collate(
        self,
        values: list[tuple[list[int], list[list[int]]]],
        /,
    ) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        row_indices = []
        item_indices = []
        labels = []
        for row_idx, (row_item_indices, row_labels) in enumerate(values):
            row_indices.extend([row_idx] * len(row_item_indices))
            item_indices.extend(row_item_indices)
            labels.extend(row_labels)
        item_indices = torch.tensor(item_indices, dtype=torch.int32)
        labels = torch.tensor(labels, dtype=torch.float32)
        return row_indices, item_indices, labels

    def save(self, path: Path, /) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "labels.json").write_text(json.dumps(self.labels))
        self.items.write_parquet(path / "items.pq")
        (path / "num_features.txt").write_text(str(self.num_features))

    @classmethod
    def load(cls, path: Path, /) -> t.Self:
        labels = json.loads((path / "labels.json").read_text())
        items = pl.read_parquet(path / "items.pq")
        num_features = int((path / "num_features.txt").read_text())
        return cls(labels, items, num_features)


class _ItemAggregator(nn.Module, ABC):
    @abstractmethod
    def forward(self, item_features: torch.Tensor, /) -> torch.Tensor: ...


class _Sum(_ItemAggregator):
    def forward(self, item_features: torch.Tensor, /) -> torch.Tensor:
        return item_features.sum(dim=1)


class _WeightedSum(_ItemAggregator):
    def __init__(self, num_features: int, /) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_features, dtype=torch.float32))

    def forward(self, item_features: torch.Tensor, /) -> torch.Tensor:
        return (item_features * self.weights.view(1, -1, 1)).sum(dim=1)


class _Concat(_ItemAggregator):
    def __init__(self, num_features: int, dim: int, /) -> None:
        super().__init__()
        self.linear = nn.Linear(num_features * dim, dim)

    def forward(self, item_features: torch.Tensor, /) -> torch.Tensor:
        return self.linear(item_features.flatten(start_dim=1))


class Layer(base.Layer):
    def __init__(
        self,
        dim: int,
        num_labels: int,
        num_features: int,
        /,
        *,
        item_aggregator: t.Literal["sum", "weighted_sum", "concat"],
    ) -> None:
        super().__init__()

        match item_aggregator:
            case "sum":
                self.item_aggregator: _ItemAggregator = _Sum()
            case "weighted_sum":
                self.item_aggregator = _WeightedSum(num_features)
            case "concat":
                self.item_aggregator = _Concat(num_features, dim)

        self.linear = nn.Linear(dim, num_labels)

    @classmethod
    def init(
        cls,
        preprocessor: Preprocessor,
        dim: int,
        /,
        *,
        item_aggregator: t.Literal["sum", "weighted_sum", "concat"] = "sum",
    ) -> t.Self:
        return cls(
            dim,
            len(preprocessor.labels),
            preprocessor.num_features,
            item_aggregator=item_aggregator,
        )

    def forward(
        self,
        backbone_embeddings: torch.Tensor,
        embedded_artifacts: tuple[list[t.Any], torch.Tensor],
        /,
    ) -> torch.Tensor:
        _, item_features = embedded_artifacts
        item_embeddings = self.item_aggregator(item_features)
        return self.linear(backbone_embeddings * item_embeddings)


class Criterion(base.Criterion):
    def __init__(self, layer: Layer) -> None:
        super().__init__()

        self._layer = [layer]

    @classmethod
    def init(cls, _preprocessor: Preprocessor, layer: Layer, /) -> t.Self:
        return cls(layer)

    def forward(
        self,
        backbone_embeddings: torch.Tensor,
        embedded_artifacts: tuple[list[t.Any], torch.Tensor],
        target: torch.Tensor,
        /,
    ) -> torch.Tensor:
        row_indices, _, labels = target
        prediction = self._layer[0](backbone_embeddings[row_indices], embedded_artifacts)
        return nn.functional.binary_cross_entropy_with_logits(prediction, labels)


class Head(base.Head):
    extras: t.ClassVar[list[str]] = ["items"]

    def __init__(
        self,
        layer: Layer,
        known_items: pl.DataFrame,
        /,
        *,
        allow_unknown_items: bool,
        label_to_weight: dict[str, float],
    ) -> None:
        self.layer = layer

        self.known_items = known_items
        self.allow_unknown_items = allow_unknown_items
        self.label_to_weight = label_to_weight

    @classmethod
    def init(
        cls,
        preprocessor: Preprocessor,
        layer: Layer,
        /,
        *,
        allow_unknown_items: bool = False,
        label_to_weight: dict[str, float] | None = None,
    ) -> t.Self:
        known_items = preprocessor.items.select(_item="item")
        label_to_weight = {
            label: 1 / len(preprocessor.labels) if label_to_weight is None else label_to_weight[label]
            for label in preprocessor.labels
        }
        return cls(
            layer,
            known_items,
            allow_unknown_items=allow_unknown_items,
            label_to_weight=label_to_weight,
        )

    def predict(
        self,
        backbone_embeddings: torch.Tensor,
        embedded_artifacts: tuple[list[t.Any], torch.Tensor],
        extras: dict[str, t.Any],
        /,
    ) -> pl.Series:
        self.layer.eval().to(backbone_embeddings.device)
        items, item_features = embedded_artifacts

        items = pl.Series("item", items, dtype=self.known_items.schema["_item"]).to_frame().with_row_index("item_index")
        if not self.allow_unknown_items:
            items = items.join(self.known_items.rename({"_item": "item"}), how="inner", on="item")

        items_to_score = (
            pl.Series("items", extras["items"], dtype=pl.List(self.known_items.schema["_item"]))
            .to_frame()
            .lazy()
            .with_row_index("row_index")
            .explode("items")
            .rename({"items": "item"})
            .with_row_index("row_item_index")
            .join(items.lazy(), how="left", on="item")
            .collect()
        )

        known_items_to_score = items_to_score.select("row_index", "row_item_index", "item_index").drop_nulls()
        if known_items_to_score.is_empty():
            return pl.Series(
                name="prediction",
                values=[[] for _ in range(len(backbone_embeddings))],
                dtype=pl.List(
                    pl.Struct(
                        {
                            "item": self.known_items.schema["_item"],
                            "probas": pl.Struct({label: pl.Float32() for label in self.label_to_weight}),
                            "score": pl.Float32(),
                        },
                    ),
                ),
            )

        row_indices = known_items_to_score["row_index"].to_torch().to(backbone_embeddings.device).long()
        item_indices = known_items_to_score["item_index"].to_torch().to(backbone_embeddings.device).long()
        backbone_embeddings = backbone_embeddings[row_indices]
        item_features = item_features[item_indices]
        known_probas = self.layer(backbone_embeddings, (None, item_features)).sigmoid()
        known_items_to_score = (
            known_items_to_score.lazy()
            .with_columns(
                **{
                    label: pl.lit(label_probas, dtype=pl.Float32())
                    for label, label_probas in zip(
                        self.label_to_weight,
                        known_probas.T.cpu().to(torch.float32).numpy(),
                        strict=True,
                    )
                },
            )
            .with_columns(
                score=pl.sum_horizontal(weight * pl.col(label) for label, weight in self.label_to_weight.items()),
            )
        )

        return (
            items_to_score.lazy()
            .join(known_items_to_score.drop("row_index", "item_index"), how="left", on="row_item_index")
            .group_by("row_index")
            .agg(
                prediction=pl.struct(
                    "item",
                    probas=pl.struct(*self.label_to_weight),
                    score="score",
                ).filter(pl.col("item_index").is_not_null()),
            )
            .sort("row_index")
            .select("prediction")
            .collect()
            .to_series()
        )

    def save(self, path: Path, /) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.layer.save(path / "layer")
        self.known_items.write_parquet(path / "known_items.pq")
        kwargs = {"allow_unknown_items": self.allow_unknown_items, "label_to_weight": self.label_to_weight}
        (path / "kwargs.json").write_text(json.dumps(kwargs))

    @classmethod
    def load(cls, path: Path, /) -> t.Self:
        layer = Layer.load(path / "layer")
        known_items = pl.read_parquet(path / "known_items.pq")
        kwargs = json.loads((path / "kwargs.json").read_text())
        return cls(layer, known_items, **kwargs)
