"""Тесты PLE-энкодера (квантильное биннинг-кодирование числовых фич)."""

import polars as pl
import torch

from perseus.core.encoders.ple import Embedder, Observer, Preprocessor


class TestObserver:
    def test_accumulates_and_drops_nulls(self) -> None:
        obs = Observer()
        obs.observe(pl.Series([1.0, 2.0, None, 3.0]))
        obs.observe(pl.Series([4.0]))
        assert sorted(obs.values.to_list()) == [1.0, 2.0, 3.0, 4.0]


class TestPreprocessor:
    def _fitted(self, values: list[float], *, max_bins: int = 4) -> Preprocessor:
        obs = Observer()
        obs.observe(pl.Series(values, dtype=pl.Float32))
        return Preprocessor.fit(obs, max_bins=max_bins)

    def test_thresholds_sorted_unique_and_span_range(self) -> None:
        pre = self._fitted([1.0, 2.0, 3.0, 4.0, 5.0], max_bins=4)
        thresholds = pre.thresholds.tolist()
        assert thresholds == sorted(thresholds)
        assert len(set(thresholds)) == len(thresholds)
        assert thresholds[0] == 1.0
        assert thresholds[-1] == 5.0

    def test_num_bins(self) -> None:
        pre = self._fitted([1.0, 2.0, 3.0, 4.0, 5.0], max_bins=4)
        assert pre.num_bins == len(pre.thresholds) - 1

    def test_static_transform_is_identity(self) -> None:
        pre = self._fitted([1.0, 2.0, 3.0])
        series = pl.Series([1.5, 2.5], dtype=pl.Float32)
        assert pre.static_transform(series).to_list() == series.to_list()

    def test_dynamic_transform_shape_and_range(self) -> None:
        pre = self._fitted([1.0, 2.0, 3.0, 4.0, 5.0], max_bins=4)
        out = pre.dynamic_transform(pl.Series([1.0, 3.0, 5.0], dtype=pl.Float32))
        assert out.shape == (3, pre.num_bins)
        assert (out >= 0.0).all()
        assert (out <= 1.0).all()

    def test_dynamic_transform_min_is_zeros_max_is_ones(self) -> None:
        pre = self._fitted([1.0, 2.0, 3.0, 4.0, 5.0], max_bins=4)
        out = pre.dynamic_transform(pl.Series([1.0, 5.0], dtype=pl.Float32))
        assert torch.allclose(out[0], torch.zeros(pre.num_bins))  # минимум → все бины 0
        assert torch.allclose(out[1], torch.ones(pre.num_bins))  # максимум → все бины 1

    def test_save_load_roundtrip(self, tmp_path) -> None:
        pre = self._fitted([1.0, 2.0, 3.0, 4.0, 5.0])
        pre.save(tmp_path)
        loaded = Preprocessor.load(tmp_path)
        assert torch.allclose(loaded.thresholds, pre.thresholds)


class TestEmbedder:
    def test_forward_shape(self) -> None:
        obs = Observer()
        obs.observe(pl.Series([1.0, 2.0, 3.0, 4.0, 5.0], dtype=pl.Float32))
        pre = Preprocessor.fit(obs, max_bins=4)
        emb = Embedder.init(pre, 8)
        out = emb(pre.dynamic_transform(pl.Series([1.0, 3.0], dtype=pl.Float32)))
        assert out.shape == (2, 8)
