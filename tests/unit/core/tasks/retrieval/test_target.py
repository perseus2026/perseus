"""Tests for retrieval target: Preprocessor, Layer, Criterion, Artifacts, Head (including predict)."""

import polars as pl
import pytest
import torch

from perseus.core.encoders.id import Embedder as IdEmbedder
from perseus.core.encoders.id import Observer as IdObserver
from perseus.core.tasks.retrieval._losses import CrossEntropy, ScalableCrossEntropy
from perseus.core.tasks.retrieval._rankers import Naive, Smmr
from perseus.core.tasks.retrieval.target import Artifacts, Criterion, Head, Layer, Preprocessor


def _fitted(*, sample_negatives: int | None = None) -> Preprocessor:
    artifacts = Artifacts(pl.DataFrame({"item": ["A", "B", "C"]}))
    values = pl.Series([["A", "B"], ["B", "C"], ["A"]])
    return Preprocessor.fit(values, artifacts, {}, sample_negatives=sample_negatives)


class TestArtifacts:
    def test_init_adds_item_column(self) -> None:
        artifacts = Artifacts(pl.DataFrame({"item": ["A", "B"]}))
        assert artifacts.items["_item"].to_list() == ["A", "B"]

    def test_static_transform_noop_and_dynamic_transform(self) -> None:
        artifacts = Artifacts(pl.DataFrame({"item": ["A", "B"]}))
        assert artifacts.static_transform({}) is artifacts
        items, features = artifacts.dynamic_transform({})
        assert items == ["A", "B"]
        assert features == {}

    def test_save_load_roundtrip(self, tmp_path) -> None:
        Artifacts(pl.DataFrame({"item": ["A", "B"]})).save(tmp_path)
        assert Artifacts.load(tmp_path).items["_item"].to_list() == ["A", "B"]


class TestPreprocessor:
    def test_fit_builds_items_with_logq_and_index(self) -> None:
        pre = _fitted()
        assert pre.items["item"].to_list() == ["A", "B", "C"]
        assert "logq" in pre.items.schema
        assert pre.items["index"].to_list() == [0, 1, 2]

    def test_static_transform_maps_items_to_indices(self) -> None:
        pre = _fitted()
        out = pre.static_transform(pl.Series([["A", "C"], ["B"]]))
        assert out.to_list() == [[0, 2], [1]]

    def test_collate_full_softmax(self) -> None:
        # sample_negatives=None -> labels over all items
        pre = _fitted()
        item_indices, labels, logq = pre.collate([[0, 1], [2]])
        assert item_indices.tolist() == [0, 1, 2]
        assert labels.tolist() == [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        assert logq.shape == (3,)

    def test_collate_with_negative_sampling(self) -> None:
        pre = _fitted(sample_negatives=1)
        item_indices, labels, logq = pre.collate([[0]])
        # positive {0} + up to 1 negative
        assert 0 in item_indices.tolist()
        assert labels.shape == (1, len(item_indices))
        assert logq.shape == (len(item_indices),)

    def test_collate_zero_negatives(self) -> None:
        pre = _fitted(sample_negatives=0)
        item_indices, labels, _ = pre.collate([[0, 1], [2]])
        assert set(item_indices.tolist()) == {0, 1, 2}
        assert labels.shape == (2, 3)

    def test_save_load_roundtrip(self, tmp_path) -> None:
        pre = _fitted(sample_negatives=5)
        pre.save(tmp_path)
        loaded = Preprocessor.load(tmp_path)
        assert loaded.sample_negatives == 5
        assert loaded.num_features == pre.num_features
        assert loaded.items["item"].to_list() == ["A", "B", "C"]


def _layer_preprocessor(num_features: int = 2) -> Preprocessor:
    return Preprocessor(pl.DataFrame({"item": ["A"], "index": [0], "logq": [0.0]}), num_features, sample_negatives=None)


class TestLayer:
    @pytest.mark.parametrize("aggregator", ["sum", "weighted_sum", "concat"])
    def test_forward_shape(self, aggregator: str) -> None:
        layer = Layer.init(_layer_preprocessor(2), 4, item_aggregator=aggregator)
        out = layer(None, (None, torch.randn(5, 2, 4)))
        assert out.shape == (5, 4)

    @pytest.mark.parametrize("aggregator", ["sum", "weighted_sum", "concat"])
    def test_save_load_roundtrip(self, aggregator: str, tmp_path) -> None:
        layer = Layer.init(_layer_preprocessor(2), 4, item_aggregator=aggregator)
        artifacts = (None, torch.randn(5, 2, 4))
        expected = layer(None, artifacts)
        layer.save(tmp_path)
        loaded = Layer.load(tmp_path)
        assert torch.allclose(loaded(None, artifacts), expected, atol=1e-6)


class TestCriterion:
    def test_init_cross_entropy(self) -> None:
        layer = Layer.init(_layer_preprocessor(), 4)
        criterion = Criterion.init(_layer_preprocessor(), layer, loss="cross_entropy")
        assert isinstance(criterion.loss, CrossEntropy)

    def test_init_scalable_cross_entropy(self) -> None:
        layer = Layer.init(_layer_preprocessor(), 4)
        criterion = Criterion.init(
            _layer_preprocessor(),
            layer,
            loss="scalable_cross_entropy",
            loss_params={"num_buckets": 2, "x_bucket_size": 2, "y_bucket_size": 2},
        )
        assert isinstance(criterion.loss, ScalableCrossEntropy)

    def test_init_unknown_loss_raises(self) -> None:
        layer = Layer.init(_layer_preprocessor(), 4)
        with pytest.raises(ValueError, match="unknown loss"):
            Criterion.init(_layer_preprocessor(), layer, loss="bogus")

    def test_forward_returns_scalar(self) -> None:
        layer = Layer.init(_layer_preprocessor(2), 8, item_aggregator="sum")
        criterion = Criterion.init(_layer_preprocessor(2), layer, loss="cross_entropy")
        # target = (item_indices, labels (B, N), logq (N,)); item_features (N, num_features, dim)
        target = (torch.arange(3), torch.eye(2, 3), torch.randn(3))
        value = criterion(torch.randn(2, 8), (None, torch.randn(3, 2, 8)), target)
        assert value.ndim == 0
        assert torch.isfinite(value)


class TestHead:
    def test_build_ranker_naive(self) -> None:
        head = Head.init(_fitted(), Layer.init(_layer_preprocessor(), 4), ranker="naive")
        assert isinstance(head._build_ranker(torch.randn(3, 4)), Naive)

    def test_build_ranker_unknown_raises(self) -> None:
        head = Head.init(_fitted(), Layer.init(_layer_preprocessor(), 4))
        head.ranker = "bogus"
        with pytest.raises(ValueError, match="unknown ranker"):
            head._build_ranker(torch.randn(3, 4))

    def test_save_load_roundtrip(self, tmp_path) -> None:
        head = Head.init(_fitted(), Layer.init(_layer_preprocessor(), 4), allow_unknown_items=True, ranker="naive")
        head.save(tmp_path)
        loaded = Head.load(tmp_path)
        assert loaded.allow_unknown_items is True
        assert loaded.ranker == "naive"
        assert loaded.known_items["_item"].to_list() == ["A", "B", "C"]


class TestArtifactsEmbed:
    def test_embed_stacks_features(self) -> None:
        artifacts = Artifacts(pl.DataFrame({"item": ["A", "B"]}))
        transformed = (["A", "B"], {"emb": torch.tensor([1, 2], dtype=torch.int32)})
        items, features = artifacts.embed(transformed, {"emb": IdEmbedder(10, 8)})
        assert items == ["A", "B"]
        assert features.shape == (2, 1, 8)  # (N, num_features, dim)

    def test_embed_with_target_selects_indices(self) -> None:
        artifacts = Artifacts(pl.DataFrame({"item": ["A", "B"]}))
        transformed = (["A", "B"], {"emb": torch.tensor([1, 2], dtype=torch.int32)})
        target = (torch.tensor([0]), None, None)
        items, features = artifacts.embed(transformed, {"emb": IdEmbedder(10, 8)}, target)
        assert items == ["A"]
        assert features.shape == (1, 1, 8)


class TestPreprocessorExtra:
    def test_dynamic_transform_passthrough(self) -> None:
        assert _fitted().dynamic_transform([0, 2]) == [0, 2]

    def test_fit_observes_artifact_features(self) -> None:
        artifacts = Artifacts(pl.DataFrame({"item": ["A", "B", "C"], "emb": ["x", "y", "z"]}))
        observer = IdObserver()
        pre = Preprocessor.fit(pl.Series([["A", "B"], ["C"]]), artifacts, {"emb": observer})
        assert pre.num_features == 1
        assert set(observer.value_counts["value"]) == {"x", "y", "z"}


def _predict_inputs(num_items: int = 3, batch: int = 2, dim: int = 8):
    backbone_embeddings = torch.randn(batch, dim)
    items = ["A", "B", "C"][:num_items]
    item_features = torch.randn(num_items, 2, dim)
    return backbone_embeddings, (items, item_features)


class TestHeadPredict:
    def test_predict_naive(self) -> None:
        pre = _fitted()
        head = Head.init(pre, Layer.init(pre, 8, item_aggregator="sum"), ranker="naive")
        backbone, embedded = _predict_inputs()
        with torch.no_grad():
            out = head.predict(backbone, embedded, {}, k=2)
        assert len(out) == 2  # one row per query
        first = out[0]
        assert len(first) == 2  # top-2
        assert all("item" in e and "score" in e for e in first)

    def test_predict_with_extra_items_only_new(self) -> None:
        pre = _fitted()
        head = Head.init(pre, Layer.init(pre, 8, item_aggregator="sum"), ranker="naive")
        backbone, embedded = _predict_inputs()
        extras = {"items": [["A"], ["B"]]}  # previously shown items per row
        with torch.no_grad():
            out = head.predict(backbone, embedded, extras, k=2, only_new=True)
        assert len(out) == 2

    def test_predict_empty_when_unknown_items(self) -> None:
        pre = _fitted()
        head = Head.init(pre, Layer.init(pre, 8, item_aggregator="sum"), allow_unknown_items=False)
        backbone = torch.randn(2, 8)
        embedded = (["Z", "Y"], torch.randn(2, 2, 8))  # items not among the known ones
        with torch.no_grad():
            out = head.predict(backbone, embedded, {}, k=2)
        assert out.to_list() == [[], []]

    def test_predict_smmr_ranker(self) -> None:
        pre = _fitted()
        head = Head.init(pre, Layer.init(pre, 8, item_aggregator="sum"), ranker="smmr")
        assert isinstance(head._build_ranker(torch.randn(3, 8)), Smmr)
        backbone, embedded = _predict_inputs()
        with torch.no_grad():
            out = head.predict(backbone, embedded, {}, k=2)
        assert len(out) == 2
