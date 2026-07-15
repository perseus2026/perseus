"""Tests for the BoW encoder (BPE tokenizer): Observer, Preprocessor, Embedder."""

import polars as pl
import torch

from perseus.core.encoders.bow import Embedder, Observer, Preprocessor


class TestObserver:
    def test_collects_unique_texts_drops_nulls(self) -> None:
        obs = Observer()
        obs.observe(pl.Series(["a b", "c", None]))
        obs.observe(pl.Series(["a b", "d"]))
        assert set(obs.texts.to_list()) == {"a b", "c", "d"}


def _fitted() -> Preprocessor:
    obs = Observer()
    obs.observe(pl.Series(["hello world", "foo bar baz", "hello foo", "world bar"]))
    return Preprocessor.fit(obs, vocab_size=50)


class TestPreprocessor:
    def test_static_transform_returns_token_id_lists(self) -> None:
        pre = _fitted()
        out = pre.static_transform(pl.Series(["hello world", "foo"]))
        assert len(out) == 2
        assert all(isinstance(ids, list) for ids in out.to_list())

    def test_dynamic_transform_pads_to_int32_tensor(self) -> None:
        pre = _fitted()
        token_ids = pre.static_transform(pl.Series(["hello world", "foo"]))
        tensor = pre.dynamic_transform(token_ids)
        assert tensor.dtype == torch.int32
        assert tensor.ndim == 2
        assert tensor.shape[0] == 2  # all rows have the same length (padding)

    def test_save_load_roundtrip(self, tmp_path) -> None:
        pre = _fitted()
        pre.save(tmp_path)
        loaded = Preprocessor.load(tmp_path)
        text = pl.Series(["hello world"])
        assert loaded.static_transform(text).to_list() == pre.static_transform(text).to_list()


class TestEmbedder:
    def test_forward_mean_pools_over_tokens(self) -> None:
        pre = _fitted()
        embedder = Embedder.init(pre, 8)
        token_ids = pre.static_transform(pl.Series(["hello world", "foo bar"]))
        tensor = pre.dynamic_transform(token_ids)
        out = embedder(tensor)
        assert out.shape == (2, 8)
        assert torch.isfinite(out).all()

    def test_padding_does_not_break_mean(self) -> None:
        # rows with different token counts → zero padding, without division by zero
        pre = _fitted()
        token_ids = pre.static_transform(pl.Series(["hello world foo bar", "foo"]))
        out = Embedder.init(pre, 8)(pre.dynamic_transform(token_ids))
        assert torch.isfinite(out).all()
