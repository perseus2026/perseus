"""Тесты history-агрегаторов backbone (forward shape).

mamba не тестируется: требует mamba_ssm (CUDA/triton), на CPU/macOS недоступен.
"""

import pytest
import torch

from perseus.core.backbone import history

DIM = 8
MAX_EVENTS = 16

CONFIGS = {
    "bert": {"num_heads": 2, "dropout": 0.0},
    "modern_bert": {"num_heads": 2, "dropout": 0.0},
    "hstu": {"num_heads": 2, "num_buckets": 16, "dropout": 0.0},
    "danet": {"dropout": 0.0},
    "ligr": {"num_heads": 2, "dropout": 0.0},
}


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings = torch.randn(2, 4, DIM)
    # в production позиции имеют dtype int32 (dataloaders/samples.py: .to_torch().int())
    positions = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 0]], dtype=torch.int32)
    timestamps = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 0.0]])
    mask = torch.zeros(2, 4, 4, dtype=torch.bool)
    return embeddings, positions, timestamps, mask


@pytest.mark.parametrize("name", ["bert", "modern_bert", "hstu", "danet", "ligr"])
def test_history_aggregator_forward_shape(name: str) -> None:
    aggregator = history.registry[name](DIM, MAX_EVENTS, **CONFIGS[name]).eval()
    out = aggregator(*_inputs())
    assert out.shape == (2, 4, DIM)
    assert torch.isfinite(out).all()
