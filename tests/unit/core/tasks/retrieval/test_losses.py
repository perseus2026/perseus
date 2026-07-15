"""Tests for retrieval losses: CrossEntropy, ScalableCrossEntropy."""

import pytest
import torch

from perseus.core.tasks.retrieval._losses import CrossEntropy, ScalableCrossEntropy


def _inputs(num_users: int = 4, num_items: int = 6, dim: int = 4):
    user_embeddings = torch.randn(num_users, dim)
    item_embeddings = torch.randn(num_items, dim)
    # each user has one positive (item i % num_items)
    labels = torch.zeros(num_users, num_items)
    for u in range(num_users):
        labels[u, u % num_items] = 1.0
    logq = torch.randn(num_items)
    return user_embeddings, item_embeddings, labels, logq


class TestCrossEntropy:
    def test_forward_scalar(self) -> None:
        loss = CrossEntropy()
        value = loss(*_inputs())
        assert value.ndim == 0
        assert torch.isfinite(value)

    def test_logq_correction_changes_value(self) -> None:
        inputs = _inputs()
        plain = CrossEntropy(logq_correction=False)(*inputs)
        corrected = CrossEntropy(logq_correction=True)(*inputs)
        assert not torch.isclose(plain, corrected)


class TestScalableCrossEntropy:
    @pytest.mark.parametrize("mix", [True, False])
    def test_forward_scalar(self, mix: bool) -> None:
        loss = ScalableCrossEntropy(num_buckets=2, x_bucket_size=2, y_bucket_size=2, mix=mix)
        value = loss(*_inputs())
        assert value.ndim == 0
        assert torch.isfinite(value)
