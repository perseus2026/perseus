import json
import typing as t
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path

import polars as pl
import torch
from torch import nn

from perseus import utils
from perseus.core import encoders
from perseus.core.tasks import base
from perseus.core.tasks.retrieval._losses import CrossEntropy, Loss, ScalableCrossEntropy
from perseus.core.tasks.retrieval._rankers import Naive, Ranker, Smmr


class Artifacts(base.Artifacts):
    def __init__(self, items: pl.DataFrame) -> None:
        if "_item" not in items.schema:
            items = items.with_columns(_item="item")  # item also can be a feature, so preserve original item
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
            item_indices, _, _ = target
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
        return cls(pl.read_parquet(path / "items.pq"))


class Preprocessor(base.Preprocessor):
    def __init__(self, items: pl.DataFrame, num_features: int, /, *, sample_negatives: int | None) -> None:
        self.items = items
        self.num_features = num_features
        self.sample_negatives = sample_negatives

    @classmethod
    def fit(
        cls,
        values: pl.Series,
        artifacts: Artifacts,
        feature_to_observer: dict[str, encoders.Observer],
        /,
        *,
        sample_negatives: int | None = None,
    ) -> t.Self:
        items = (
            values.explode()
            .drop_nulls()
            .rename("item")
            .value_counts()
            .lazy()
            .select("item", logq=(pl.col("count") / pl.col("count").sum()).log())
            .sort("item")
            .with_row_index()
            .collect()
        )
        artifacts.items = artifacts.items.join(items.select("item"), how="inner", on="item").sort("item")
        for feature, observer in feature_to_observer.items():
            observer.observe(artifacts.items[feature])
        return cls(items, len(feature_to_observer), sample_negatives=sample_negatives)

    def static_transform(self, values: pl.Series, /) -> pl.Series:
        return (
            values.to_frame("items")
            .lazy()
            .with_row_index("row_index")
            .explode("items")
            .rename({"items": "item"})
            .join(self.items.lazy().select("item", item_index="index"), how="left", on="item", maintain_order="left")
            .group_by("row_index", maintain_order=True)
            .agg(item_indices="item_index")
            .select("item_indices")
            .collect()
            .to_series()
        )

    def dynamic_transform(self, value: list[int], /) -> list[int]:
        return value

    def collate(self, values: list[list[int]], /) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.sample_negatives is None:
            item_indices = self._item_indices
            logq = self._logq

            labels_row_indices = []
            labels_col_indices = []
            for row_idx, row_item_indices in enumerate(values):
                labels_row_indices.extend([row_idx] * len(row_item_indices))
                labels_col_indices.extend(row_item_indices)
            labels = torch.sparse_coo_tensor(
                indices=[labels_row_indices, labels_col_indices],
                values=[1] * len(labels_row_indices),
                size=(len(values), len(item_indices)),
                dtype=torch.float32,
            )
        else:
            uniq_item_indices = {item_index for item_indices in values for item_index in item_indices}
            if self.sample_negatives > 0:
                uniq_item_indices |= set(
                    [
                        item_index
                        for item_index in torch.randperm(len(self.items))[
                            : self.sample_negatives + len(uniq_item_indices)
                        ].tolist()
                        if item_index not in uniq_item_indices
                    ][: self.sample_negatives],
                )

            item_indices = torch.tensor(list(uniq_item_indices), dtype=torch.int32)
            item_index_to_label_index = {
                item_index: label_index for label_index, item_index in enumerate(uniq_item_indices)
            }

            labels_row_indices = []
            labels_col_indices = []
            for row_idx, row_item_indices in enumerate(values):
                labels_row_indices.extend([row_idx] * len(row_item_indices))
                labels_col_indices.extend([item_index_to_label_index[item_index] for item_index in row_item_indices])
            labels = torch.sparse_coo_tensor(
                indices=[labels_row_indices, labels_col_indices],
                values=[1] * len(labels_row_indices),
                size=(len(values), len(item_indices)),
                dtype=torch.float32,
            )

            logq = self._logq[item_indices]

        return item_indices, labels.to_dense(), logq

    def save(self, path: Path, /) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.items.write_parquet(path / "items.pq")
        (path / "num_features.txt").write_text(str(self.num_features))
        (path / "kwargs.json").write_text(json.dumps({"sample_negatives": self.sample_negatives}))

    @classmethod
    def load(cls, path: Path, /) -> t.Self:
        items = pl.read_parquet(path / "items.pq")
        num_features = int((path / "num_features.txt").read_text())
        kwargs = json.loads((path / "kwargs.json").read_text())
        return cls(items, num_features, **kwargs)

    @cached_property
    def _item_indices(self) -> torch.Tensor:
        return self.items["index"].to_torch()

    @cached_property
    def _logq(self) -> torch.Tensor:
        return self.items["logq"].to_torch()


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

    @classmethod
    def init(
        cls,
        preprocessor: Preprocessor,
        dim: int,
        /,
        *,
        item_aggregator: t.Literal["sum", "weighted_sum", "concat"] = "sum",
    ) -> t.Self:
        return cls(dim, preprocessor.num_features, item_aggregator=item_aggregator)

    def forward(
        self,
        _backbone_embeddings: torch.Tensor,
        embedded_artifacts: tuple[list[t.Any], torch.Tensor],
        /,
    ) -> torch.Tensor:
        _, item_features = embedded_artifacts
        return self.item_aggregator(item_features)


class Criterion(base.Criterion):
    def __init__(self, layer: Layer, /, *, loss: Loss) -> None:
        super().__init__()

        self._layer = [layer]
        self.loss = loss

    @classmethod
    def init(
        cls,
        _preprocessor: Preprocessor,
        layer: Layer,
        /,
        *,
        loss: t.Literal["cross_entropy", "scalable_cross_entropy"] = "cross_entropy",
        loss_params: dict[str, t.Any] | None = None,
    ) -> t.Self:
        match loss:
            case "cross_entropy":
                loss_cls = CrossEntropy
            case "scalable_cross_entropy":
                loss_cls = ScalableCrossEntropy
            case _:
                raise ValueError(f"unknown loss = {loss}")
        loss_obj = loss_cls(**(loss_params or {}))
        return cls(layer, loss=loss_obj)

    def forward(
        self,
        backbone_embeddings: torch.Tensor,
        embedded_artifacts: tuple[list[t.Any], torch.Tensor],
        target: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        /,
    ) -> torch.Tensor:
        _, labels, logq = target
        item_embeddings = self._layer[0](backbone_embeddings, embedded_artifacts)
        return self.loss(backbone_embeddings, item_embeddings, labels, logq)


class Head(base.Head):
    extras: t.ClassVar[list[str]] = ["items"]

    def __init__(
        self,
        layer: Layer,
        known_items: pl.DataFrame,
        /,
        *,
        allow_unknown_items: bool,
        ranker: t.Literal["naive", "smmr"],
        ranker_params: dict[str, t.Any] | None,
    ) -> None:
        self.layer = layer
        self.known_items = known_items
        self.allow_unknown_items = allow_unknown_items
        self.ranker = ranker
        self.ranker_params = ranker_params

    @classmethod
    def init(
        cls,
        preprocessor: Preprocessor,
        layer: Layer,
        /,
        *,
        allow_unknown_items: bool = False,
        ranker: t.Literal["naive", "smmr"] = "naive",
        ranker_params: dict[str, t.Any] | None = None,
    ) -> t.Self:
        return cls(
            layer,
            preprocessor.items.select(_item="item"),
            allow_unknown_items=allow_unknown_items,
            ranker=ranker,
            ranker_params=ranker_params,
        )

    def predict(
        self,
        backbone_embeddings: torch.Tensor,
        embedded_artifacts: tuple[list[t.Any], torch.Tensor],
        extras: dict[str, t.Any],
        /,
        *,
        k: int,
        only_new: bool = False,
    ) -> pl.Series:
        self.layer.eval().to(backbone_embeddings.device)
        items, item_features = embedded_artifacts

        items = pl.Series("item", items, dtype=self.known_items.schema["_item"]).to_frame().with_row_index("item_index")
        if not self.allow_unknown_items:
            items = items.join(self.known_items.rename({"_item": "item"}), how="inner", on="item")
        if items.is_empty():
            return pl.Series(
                "prediction",
                [[] for _ in range(len(backbone_embeddings))],
                dtype=pl.List(pl.Struct({"item": self.known_items.schema["_item"], "score": pl.Float32()})),
            )

        item_indices = items["item_index"].to_torch().to(backbone_embeddings.device).long()
        item_features = item_features[item_indices]
        item_embeddings = self.layer(backbone_embeddings, (None, item_features))
        logits = backbone_embeddings @ item_embeddings.T
        items = items.drop("item_index").with_row_index("item_index")

        extra_prediction = None
        if (extra_items := extras.get("items")) is not None:
            extra_items = (
                pl.Series("items", extra_items, dtype=pl.List(self.known_items.schema["_item"]))
                .to_frame()
                .lazy()
                .with_row_index("index")
                .explode("items")
                .rename({"items": "item"})
                .join(items.lazy(), how="inner", on="item", maintain_order="left")
                .collect()
            )
            extra_scores = []
            for (index,), index_extra_items in extra_items.group_by("index", maintain_order=True):
                index_extra_item_indices = index_extra_items["item_index"].to_list()
                extra_scores.extend(logits[index, index_extra_item_indices].tolist())
                if only_new:
                    logits[index, index_extra_item_indices] = -torch.inf
            extra_prediction = extra_items.select(
                "index",
                "item",
                score=pl.Series(values=extra_scores, dtype=pl.Float32()),
            )

        ranker = self._build_ranker(item_embeddings)
        top_indices, top_logits = ranker.top(logits, k=min(k, len(items)))
        prediction = (
            pl.LazyFrame(
                {
                    "item_index": pl.Series(values=top_indices.tolist(), dtype=pl.List(pl.UInt32())),
                    "score": pl.Series(values=top_logits.tolist(), dtype=pl.List(pl.Float32())),
                },
            )
            .with_row_index("index")
            .explode("item_index", "score")
            .join(items.lazy(), how="inner", on="item_index", maintain_order="left")
            .select("index", "item", "score")
            .collect()
        )

        if extra_prediction is not None:
            prediction = (
                pl.concat([prediction.lazy(), extra_prediction.lazy()])
                .group_by("index", "item")
                .agg(pl.col("score").first())
                .sort("index", "score", "item", descending=[False, True, False])
                .collect()
            )

        return (
            pl.int_range(len(backbone_embeddings), dtype=pl.UInt32(), eager=True)
            .to_frame("index")
            .lazy()
            .join(
                prediction.lazy().group_by("index").agg(prediction=pl.struct("item", "score")),
                how="left",
                on="index",
                maintain_order="left",
            )
            .select(pl.col("prediction").fill_null([]))
            .collect()
            .to_series()
        )

    def save(self, path: Path, /) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.layer.save(path / "layer")
        self.known_items.write_parquet(path / "known_items.pq")
        kwargs = {
            "allow_unknown_items": self.allow_unknown_items,
            "ranker": self.ranker,
            "ranker_params": self.ranker_params,
        }
        (path / "kwargs.json").write_text(json.dumps(kwargs))

    @classmethod
    def load(cls, path: Path, /) -> t.Self:
        layer = Layer.load(path / "layer")
        known_items = pl.read_parquet(path / "known_items.pq")
        kwargs = json.loads((path / "kwargs.json").read_text())
        return cls(layer, known_items, **kwargs)

    def _build_ranker(self, item_embeddings: torch.Tensor, /) -> Ranker:
        match self.ranker:
            case "naive":
                return Naive(item_embeddings)
            case "smmr":
                return Smmr(item_embeddings, **(self.ranker_params or {}))
            case _:
                raise ValueError(f"unknown ranker = {self.ranker}")
