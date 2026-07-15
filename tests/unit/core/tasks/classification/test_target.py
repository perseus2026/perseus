"""Тесты classification target: Preprocessor, Layer, Criterion, Head, Artifacts."""

import polars as pl
import pytest
import torch

from perseus.core.tasks.classification.target import Artifacts, Criterion, Head, Layer, Preprocessor


def _fit() -> Preprocessor:
    return Preprocessor.fit(pl.Series(["b", "a", "c", "a"]), Artifacts(), {})


class TestPreprocessor:
    def test_fit_builds_sorted_label_index(self) -> None:
        pre = _fit()
        assert pre.label_to_index == {"a": 0, "b": 1, "c": 2}

    def test_static_transform_maps_labels(self) -> None:
        out = _fit().static_transform(pl.Series(["a", "c", "b"]))
        assert out.to_list() == [0, 2, 1]
        assert out.dtype == pl.UInt32

    def test_static_transform_unknown_label_is_null(self) -> None:
        out = _fit().static_transform(pl.Series(["a", "zzz"]))
        assert out.to_list() == [0, None]

    def test_collate_to_long_tensor(self) -> None:
        tensor = _fit().collate([0, 1, 2])
        assert tensor.dtype == torch.long
        assert tensor.tolist() == [0, 1, 2]

    def test_save_load_roundtrip(self, tmp_path) -> None:
        pre = _fit()
        pre.save(tmp_path)
        loaded = Preprocessor.load(tmp_path)
        assert loaded.label_to_index == pre.label_to_index


class TestLayer:
    def test_forward_shape_matches_num_classes(self) -> None:
        layer = Layer.init(_fit(), 4)  # 3 класса
        out = layer(torch.randn(2, 4), None)
        assert out.shape == (2, 3)

    def test_save_load_roundtrip(self, tmp_path) -> None:
        layer = Layer.init(_fit(), 4)
        x = torch.randn(2, 4)
        expected = layer(x, None)
        layer.save(tmp_path)
        loaded = Layer.load(tmp_path)
        assert torch.allclose(loaded(x, None), expected)


class TestCriterion:
    def test_forward_cross_entropy_scalar(self) -> None:
        layer = Layer.init(_fit(), 4)
        criterion = Criterion.init(_fit(), layer)
        target = torch.tensor([0, 2], dtype=torch.long)
        value = criterion(torch.randn(2, 4), None, target)
        assert value.ndim == 0
        assert value.item() > 0


class TestHead:
    def test_predict_returns_probability_struct(self) -> None:
        pre = _fit()
        head = Head.init(pre, Layer.init(pre, 4))
        out = head.predict(torch.randn(5, 4), None, {})
        assert out.dtype == pl.Struct({"a": pl.Float32, "b": pl.Float32, "c": pl.Float32})
        # softmax → вероятности по строке суммируются в 1
        first = out[0]
        assert sum(first.values()) == pytest.approx(1.0, abs=1e-5)

    def test_save_load_roundtrip(self, tmp_path) -> None:
        pre = _fit()
        head = Head.init(pre, Layer.init(pre, 4))
        x = torch.randn(2, 4)
        expected = head.predict(x, None, {})
        head.save(tmp_path)
        loaded = Head.load(tmp_path)
        assert loaded.labels == head.labels
        assert loaded.predict(x, None, {}).struct.field("a").to_list() == pytest.approx(
            expected.struct.field("a").to_list(),
        )


def test_artifacts_trivial_methods(tmp_path) -> None:
    artifacts = Artifacts()
    assert artifacts.static_transform({}) is artifacts
    assert artifacts.dynamic_transform({}) is None
    assert artifacts.embed(None, {}) is None
    assert artifacts.save(tmp_path) is None
    assert isinstance(Artifacts.load(tmp_path), Artifacts)
