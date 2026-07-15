"""Тесты ID-энкодера: Observer, Preprocessor, Embedder."""

import polars as pl
import torch

from perseus.core.encoders.id import Embedder, Observer, Preprocessor


class TestObserver:
    def test_counts_values(self) -> None:
        obs = Observer()
        obs.observe(pl.Series(["a", "b", "a"]))
        counts = dict(zip(obs.value_counts["value"], obs.value_counts["count"], strict=True))
        assert counts == {"a": 2, "b": 1}

    def test_merges_across_calls_and_drops_nulls(self) -> None:
        obs = Observer()
        obs.observe(pl.Series(["a", "b", None, "a"]))
        obs.observe(pl.Series(["a", "c"]))
        counts = dict(zip(obs.value_counts["value"], obs.value_counts["count"], strict=True))
        assert counts == {"a": 3, "b": 1, "c": 1}


class TestPreprocessor:
    def _fitted(self, **kwargs) -> Preprocessor:
        obs = Observer()
        obs.observe(pl.Series(["a", "a", "a", "b", "b", "c"]))
        return Preprocessor.fit(obs, **kwargs)

    def test_fit_assigns_one_based_indices_sorted_by_count(self) -> None:
        pre = self._fitted()
        mapping = dict(zip(pre.values["value"], pre.values["index"], strict=True))
        # сортировка по count desc, value asc → a(3)=1, b(2)=2, c(1)=3
        assert mapping == {"a": 1, "b": 2, "c": 3}
        assert pre.cardinality == 3
        assert pre.has_unk is False

    def test_top_k_filters_and_sets_has_unk(self) -> None:
        pre = self._fitted(top_k=2)
        assert set(pre.values["value"]) == {"a", "b"}
        assert pre.has_unk is True

    def test_min_count_filters(self) -> None:
        pre = self._fitted(min_count=2)
        assert set(pre.values["value"]) == {"a", "b"}
        assert pre.has_unk is True

    def test_static_transform_maps_known_and_unknown(self) -> None:
        pre = self._fitted()
        out = pre.static_transform(pl.Series(["a", "c", "zzz"]))
        # a→1, c→3, неизвестное→cardinality+1=4
        assert out.to_list() == [1, 3, 4]

    def test_dynamic_transform_to_int32_tensor(self) -> None:
        pre = self._fitted()
        tensor = pre.dynamic_transform(pl.Series([1, 2, 3]))
        assert tensor.dtype == torch.int32
        assert tensor.tolist() == [1, 2, 3]

    def test_save_load_roundtrip(self, tmp_path) -> None:
        pre = self._fitted(top_k=2)
        pre.save(tmp_path)
        loaded = Preprocessor.load(tmp_path)
        assert loaded.has_unk is True
        assert loaded.values.equals(pre.values)


class TestEmbedder:
    def _embedder(self, dim: int = 4, **kwargs) -> Embedder:
        obs = Observer()
        obs.observe(pl.Series(["a", "a", "b", "c"]))
        pre = Preprocessor.fit(obs, **kwargs)
        return Embedder.init(pre, dim)

    def test_forward_shape(self) -> None:
        emb = self._embedder(dim=4)
        out = emb(torch.tensor([0, 1, 2, 3], dtype=torch.int32))
        assert out.shape == (4, 4)

    def test_padding_index_is_zero(self) -> None:
        emb = self._embedder(dim=4)
        out = emb(torch.tensor([0], dtype=torch.int32))
        assert torch.allclose(out[0], torch.zeros(4))

    def test_unknown_index_maps_to_unknown_embedding(self) -> None:
        emb = self._embedder(dim=4)
        # cardinality=3, num_embeddings=4 → индекс 4 трактуется как unknown
        out = emb(torch.tensor([4], dtype=torch.int32))
        assert torch.allclose(out[0], emb.unknown_embedding, atol=1e-6)
