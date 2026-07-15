"""Тесты ranking target: Preprocessor, Layer (агрегаторы), Criterion, Head, Artifacts."""

import polars as pl
import pytest
import torch

from perseus.core.tasks.ranking.target import Artifacts, Criterion, Head, Layer, Preprocessor


def _preprocessor(num_features: int = 2, labels: list[str] | None = None) -> Preprocessor:
    labels = labels or ["click", "buy"]
    items = pl.DataFrame({"item": ["A", "B"], "index": [0, 1]})
    return Preprocessor(labels, items, num_features)


def _embedded_artifacts(n: int = 5, num_features: int = 2, dim: int = 4) -> tuple[None, torch.Tensor]:
    return None, torch.randn(n, num_features, dim)


class TestLayer:
    @pytest.mark.parametrize("aggregator", ["sum", "weighted_sum", "concat"])
    def test_forward_shape(self, aggregator: str) -> None:
        layer = Layer.init(_preprocessor(), 4, item_aggregator=aggregator)
        out = layer(torch.randn(5, 4), _embedded_artifacts(5, 2, 4))
        assert out.shape == (5, 2)  # 2 метки (click, buy)

    @pytest.mark.parametrize("aggregator", ["sum", "weighted_sum", "concat"])
    def test_save_load_roundtrip(self, aggregator: str, tmp_path) -> None:
        layer = Layer.init(_preprocessor(), 4, item_aggregator=aggregator)
        backbone = torch.randn(5, 4)
        artifacts = _embedded_artifacts(5, 2, 4)
        expected = layer(backbone, artifacts)
        layer.save(tmp_path)
        loaded = Layer.load(tmp_path)
        assert torch.allclose(loaded(backbone, artifacts), expected, atol=1e-6)


class TestCriterion:
    def test_forward_bce_scalar(self) -> None:
        layer = Layer.init(_preprocessor(), 4, item_aggregator="sum")
        criterion = Criterion.init(_preprocessor(), layer)
        # target = (row_indices, item_indices, labels); labels — (N, num_labels) float
        row_indices = [0, 1, 2, 3, 4]
        labels = torch.randint(0, 2, (5, 2)).float()
        target = (row_indices, torch.arange(5), labels)
        value = criterion(torch.randn(5, 4), _embedded_artifacts(5, 2, 4), target)
        assert value.ndim == 0
        assert torch.isfinite(value)


class TestArtifacts:
    def test_init_adds_item_column(self) -> None:
        artifacts = Artifacts(pl.DataFrame({"item": ["A", "B"]}))
        assert "_item" in artifacts.items.schema
        assert artifacts.items["_item"].to_list() == ["A", "B"]

    def test_init_preserves_existing_item_column(self) -> None:
        items = pl.DataFrame({"item": ["A"], "_item": ["orig"]})
        artifacts = Artifacts(items)
        assert artifacts.items["_item"].to_list() == ["orig"]

    def test_static_transform_no_preprocessors_is_noop(self) -> None:
        artifacts = Artifacts(pl.DataFrame({"item": ["A", "B"]}))
        assert artifacts.static_transform({}) is artifacts

    def test_dynamic_transform_returns_items_and_features(self) -> None:
        artifacts = Artifacts(pl.DataFrame({"item": ["A", "B"]}))
        items, features = artifacts.dynamic_transform({})
        assert items == ["A", "B"]
        assert features == {}

    def test_save_load_roundtrip(self, tmp_path) -> None:
        artifacts = Artifacts(pl.DataFrame({"item": ["A", "B"]}))
        artifacts.save(tmp_path)
        loaded = Artifacts.load(tmp_path)
        assert loaded.items["_item"].to_list() == ["A", "B"]


def _target_values() -> pl.Series:
    # List(Struct(item, labels: Struct(click, buy)))
    return pl.Series(
        [
            [
                {"item": "A", "labels": {"click": 1, "buy": 0}},
                {"item": "B", "labels": {"click": 0, "buy": 1}},
            ],
            [{"item": "A", "labels": {"click": 1, "buy": 1}}],
        ],
    )


class TestPreprocessorFit:
    def test_fit_collects_labels_and_items(self) -> None:
        artifacts = Artifacts(pl.DataFrame({"item": ["A", "B"]}))
        pre = Preprocessor.fit(_target_values(), artifacts, {})
        assert pre.labels == ["click", "buy"]
        assert pre.items["item"].to_list() == ["A", "B"]
        assert pre.items["index"].to_list() == [0, 1]
        assert pre.num_features == 0

    def test_static_transform_builds_labeled_item_indices(self) -> None:
        artifacts = Artifacts(pl.DataFrame({"item": ["A", "B"]}))
        pre = Preprocessor.fit(_target_values(), artifacts, {})
        out = pre.static_transform(_target_values())
        first = out[0]  # элемент struct-серии → dict
        assert list(first["item_indices"]) == [0, 1]
        # метки click,buy для A=[1,0], B=[0,1]
        assert [list(arr) for arr in first["labels"]] == [[1, 0], [0, 1]]

    def test_dynamic_transform_unpacks_struct(self) -> None:
        pre = Preprocessor.fit(_target_values(), Artifacts(pl.DataFrame({"item": ["A", "B"]})), {})
        value = {"item_indices": [0, 1], "labels": [[1, 0], [0, 1]]}
        assert pre.dynamic_transform(value) == ([0, 1], [[1, 0], [0, 1]])

    def test_collate_builds_row_item_label_tensors(self) -> None:
        pre = Preprocessor.fit(_target_values(), Artifacts(pl.DataFrame({"item": ["A", "B"]})), {})
        row_indices, item_indices, labels = pre.collate(
            [([0, 1], [[1, 0], [0, 1]]), ([0], [[1, 1]])],
        )
        assert row_indices == [0, 0, 1]
        assert item_indices.tolist() == [0, 1, 0]
        assert item_indices.dtype == torch.int32
        assert labels.tolist() == [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

    def test_save_load_roundtrip(self, tmp_path) -> None:
        pre = Preprocessor.fit(_target_values(), Artifacts(pl.DataFrame({"item": ["A", "B"]})), {})
        pre.save(tmp_path)
        loaded = Preprocessor.load(tmp_path)
        assert loaded.labels == ["click", "buy"]
        assert loaded.num_features == 0
        assert loaded.items["item"].to_list() == ["A", "B"]


class TestHead:
    def test_init_default_label_weights(self) -> None:
        head = Head.init(_preprocessor(), Layer.init(_preprocessor(), 8, item_aggregator="sum"))
        assert head.label_to_weight == {"click": 0.5, "buy": 0.5}

    def test_init_custom_label_weights(self) -> None:
        head = Head.init(
            _preprocessor(),
            Layer.init(_preprocessor(), 8, item_aggregator="sum"),
            label_to_weight={"click": 0.7, "buy": 0.3},
        )
        assert head.label_to_weight == {"click": 0.7, "buy": 0.3}

    def test_predict_scores_items_per_row(self) -> None:
        pre = _preprocessor(num_features=2)  # items A,B; labels click,buy
        head = Head.init(pre, Layer.init(pre, 8, item_aggregator="sum"))
        backbone_embeddings = torch.randn(2, 8)
        embedded_artifacts = (["A", "B"], torch.randn(2, 2, 8))
        extras = {"items": [["A", "B"], ["A"]]}
        with torch.no_grad():
            out = head.predict(backbone_embeddings, embedded_artifacts, extras)
        assert len(out) == 2
        row0 = out[0]
        assert {entry["item"] for entry in row0} == {"A", "B"}
        assert all("probas" in entry and "score" in entry for entry in row0)

    def test_save_load_roundtrip(self, tmp_path) -> None:
        pre = _preprocessor(num_features=2)
        head = Head.init(pre, Layer.init(pre, 8, item_aggregator="sum"), allow_unknown_items=True)
        head.save(tmp_path)
        loaded = Head.load(tmp_path)
        assert loaded.allow_unknown_items is True
        assert loaded.label_to_weight == {"click": 0.5, "buy": 0.5}
        assert loaded.known_items["_item"].to_list() == ["A", "B"]
