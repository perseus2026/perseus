"""Tests for backbone event and context aggregators + registries."""

import pytest
import torch

from perseus.core.backbone import context, event, history


class TestEventAggregators:
    @pytest.mark.parametrize("name", ["sum", "weighted_sum", "concat"])
    def test_forward_reduces_feature_dim(self, name: str) -> None:
        # input (B, L, num_features, dim) → output (B, L, dim)
        aggregator = event.registry[name](4, 2)
        out = aggregator(torch.randn(2, 3, 2, 4))
        assert out.shape == (2, 3, 4)

    def test_sum_is_feature_sum(self) -> None:
        aggregator = event.registry["sum"](4, 2)
        features = torch.randn(2, 3, 2, 4)
        assert torch.allclose(aggregator(features), features.sum(dim=2))


class TestContextAggregator:
    def test_identity_returns_input(self) -> None:
        aggregator = context.registry["identity"](4, 2)
        features = torch.randn(2, 1, 4)
        assert torch.allclose(aggregator(features), features)


def test_registries_have_expected_keys() -> None:
    assert set(event.registry) == {"sum", "weighted_sum", "concat"}
    assert set(context.registry) == {"identity"}
    assert {"bert", "modern_bert", "hstu", "mamba", "danet", "ligr"} <= set(history.registry)
