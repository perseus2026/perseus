"""Тесты regression target: Preprocessor, Layer, Criterion, Head, Artifacts."""

import polars as pl
import pytest
import torch

from perseus.core.tasks.regression.target import Artifacts, Criterion, Head, Layer, Preprocessor


def _fit(scaler: str = "none", **params) -> Preprocessor:
    values = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    return Preprocessor.fit(values, Artifacts(), {}, scaler=scaler, scaler_params=params or None)


class TestPreprocessor:
    @pytest.mark.parametrize("scaler", ["none", "log", "standard", "min-max", "max-abs", "robust"])
    def test_fit_with_scalers(self, scaler: str) -> None:
        pre = _fit(scaler)
        out = pre.static_transform(pl.Series([1.0, 2.0, 3.0]))
        assert out.dtype == pl.Float32
        assert len(out) == 3

    def test_fit_unknown_scaler_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown scaler"):
            _fit("bogus")

    def test_standard_scaler_centers_data(self) -> None:
        pre = _fit("standard")
        # среднее обучающей выборки (3.0) → ~0 после стандартизации
        assert pre.static_transform(pl.Series([3.0]))[0] == pytest.approx(0.0, abs=1e-5)

    def test_dynamic_transform_is_identity(self) -> None:
        assert _fit().dynamic_transform(2.5) == 2.5

    def test_collate_to_float_tensor(self) -> None:
        tensor = _fit().collate([1.0, 2.0])
        assert tensor.dtype == torch.float32
        assert tensor.tolist() == [1.0, 2.0]

    def test_save_load_roundtrip(self, tmp_path) -> None:
        pre = _fit("standard")
        pre.save(tmp_path)
        loaded = Preprocessor.load(tmp_path)
        assert loaded.static_transform(pl.Series([3.0]))[0] == pytest.approx(
            pre.static_transform(pl.Series([3.0]))[0],
        )


class TestLayer:
    def test_forward_shape(self) -> None:
        layer = Layer.init(_fit(), 4)
        out = layer(torch.randn(3, 4), None)
        assert out.shape == (3, 1)

    def test_save_load_roundtrip(self, tmp_path) -> None:
        layer = Layer.init(_fit(), 4)
        x = torch.randn(2, 4)
        expected = layer(x, None)
        layer.save(tmp_path)
        loaded = Layer.load(tmp_path)
        assert torch.allclose(loaded(x, None), expected)


class TestCriterion:
    @pytest.mark.parametrize("loss", ["mse", "l1", "smooth_l1", "huber"])
    def test_forward_returns_scalar(self, loss: str) -> None:
        layer = Layer.init(_fit(), 4)
        criterion = Criterion.init(_fit(), layer, loss=loss)
        value = criterion(torch.randn(3, 4), None, torch.randn(3))
        assert value.ndim == 0
        assert torch.isfinite(value)

    def test_unknown_loss_raises(self) -> None:
        layer = Layer.init(_fit(), 4)
        with pytest.raises(ValueError, match="unknown loss"):
            Criterion.init(_fit(), layer, loss="bogus")


class TestHead:
    # predict вызывается в production под @torch.no_grad (см. make_head_predictions),
    # regression-версия зовёт .numpy() и требует этого контекста — воспроизводим его.
    def test_predict_returns_float_series(self) -> None:
        pre = _fit("standard")
        head = Head.init(pre, Layer.init(pre, 4))
        with torch.no_grad():
            out = head.predict(torch.randn(3, 4), None, {})
        assert out.dtype == pl.Float32
        assert len(out) == 3

    def test_save_load_roundtrip(self, tmp_path) -> None:
        pre = _fit("standard")
        head = Head.init(pre, Layer.init(pre, 4))
        x = torch.randn(2, 4)
        with torch.no_grad():
            expected = head.predict(x, None, {})
            head.save(tmp_path)
            loaded = Head.load(tmp_path)
            assert loaded.predict(x, None, {}).to_list() == pytest.approx(expected.to_list())


def test_artifacts_trivial_methods(tmp_path) -> None:
    artifacts = Artifacts()
    assert artifacts.static_transform({}) is artifacts
    assert artifacts.dynamic_transform({}) is None
    assert artifacts.embed(None, {}) is None
    assert artifacts.save(tmp_path) is None
    assert isinstance(Artifacts.load(tmp_path), Artifacts)
