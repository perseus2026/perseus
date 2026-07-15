"""Tests for retrieval rankers: Naive, Smmr, _softmax."""

import pytest
import torch

from perseus.core.tasks.retrieval._rankers import Naive, Smmr, _softmax


def test_naive_returns_topk() -> None:
    logits = torch.tensor([[0.1, 0.9, 0.5, 0.2], [0.4, 0.3, 0.8, 0.1]])
    ranker = Naive(torch.randn(4, 3))
    indices, values = ranker.top(logits, k=2)
    assert indices.shape == (2, 2)
    assert values.shape == (2, 2)
    # top-1 per row: item 1 and item 2
    assert indices[:, 0].tolist() == [1, 2]


def test_softmax_normalizes() -> None:
    out = _softmax(torch.tensor([[1.0, 2.0, 3.0]]))
    assert out.sum().item() == pytest.approx(1.0)
    assert (out >= 0).all()


class TestSmmr:
    def test_top_shapes(self) -> None:
        item_embeddings = torch.randn(5, 4)
        logits = torch.randn(2, 5)
        indices, values = Smmr(item_embeddings, pool_size=5).top(logits, k=3)
        assert indices.shape == (2, 3)
        assert values.shape == (2, 3)
        # selected indices are valid and unique within a row
        for row in indices.tolist():
            assert len(set(row)) == len(row)

    def test_k_greater_than_pool_raises(self) -> None:
        with pytest.raises(ValueError, match="lower than pool size"):
            Smmr(torch.randn(5, 4), pool_size=2).top(torch.randn(2, 5), k=3)
