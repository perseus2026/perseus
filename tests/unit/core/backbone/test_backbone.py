"""Тесты Backbone: сборка агрегаторов, forward (с контекстом и без), helper-методы, save/load."""

import torch

from perseus.core.backbone import Backbone
from perseus.core.encoders.id import Embedder

DIM = 8
MAX_EVENTS = 16


def _embedders() -> dict[str, Embedder]:
    return {"event": Embedder(10, DIM), "city": Embedder(10, DIM)}


def _backbone(*, with_context: bool = False) -> Backbone:
    return Backbone(
        DIM,
        history_aggregator_config={
            "type": "bert",
            "max_events_per_sequence": MAX_EVENTS,
            "params": {"num_heads": 2, "dropout": 0.0},
        },
        event_aggregator_config={"type": "sum", "num_features": 1, "params": {}},
        context_aggregator_config=({"type": "identity", "num_features": 1, "params": {}} if with_context else None),
    )


def _events() -> tuple:
    events_features = {"event": torch.randint(1, 5, (2, 3), dtype=torch.int32)}
    positions = torch.tensor([[1, 2, 3], [1, 2, 0]], dtype=torch.int32)
    timestamps = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 0.0]])
    sample_timestamp = torch.tensor([4.0, 3.0])
    return events_features, positions, timestamps, sample_timestamp


def test_init_builds_components() -> None:
    backbone = _backbone()
    assert hasattr(backbone, "history_aggregator")
    assert hasattr(backbone, "event_aggregator")
    assert backbone.readout_token.shape == (DIM,)
    assert not hasattr(backbone, "context_aggregator")


def test_init_with_context_builds_context_aggregator() -> None:
    assert hasattr(_backbone(with_context=True), "context_aggregator")


def test_forward_without_context() -> None:
    backbone = _backbone().eval()
    out = backbone(_embedders(), *_events())
    assert out.shape == (2, DIM)
    assert torch.isfinite(out).all()


def test_forward_with_context() -> None:
    backbone = _backbone(with_context=True).eval()
    events_features, positions, timestamps, sample_timestamp = _events()
    context_features = {"city": torch.randint(1, 5, (2,), dtype=torch.int32)}
    out = backbone(_embedders(), events_features, positions, timestamps, sample_timestamp, context_features)
    assert out.shape == (2, DIM)
    assert torch.isfinite(out).all()


class TestHelpers:
    def test_build_positions_without_context(self) -> None:
        backbone = _backbone()
        positions = torch.tensor([[1, 2, 3], [1, 2, 0]], dtype=torch.int32)
        out = backbone._build_positions(positions, 0)
        # добавляется 1 readout-позиция (нулевая)
        assert out.shape == (2, 4)
        assert out[:, -1].tolist() == [0, 0]

    def test_build_positions_with_context(self) -> None:
        backbone = _backbone()
        positions = torch.tensor([[1, 2, 3]], dtype=torch.int32)
        out = backbone._build_positions(positions, 2)
        # 2 context + 1 readout = 3 дополнительных нулевых позиции
        assert out.shape == (1, 6)

    def test_build_timestamps(self) -> None:
        backbone = _backbone()
        timestamps = torch.tensor([[1.0, 2.0, 3.0]])
        sample_timestamp = torch.tensor([9.0])
        out = backbone._build_timestamps(timestamps, sample_timestamp, 0)
        assert out.shape == (1, 4)
        assert out[0, -1].item() == 9.0  # readout-таймстемп = таймстемп сэмпла

    def test_build_mask_shape_and_diagonal(self) -> None:
        backbone = _backbone()
        positions = torch.tensor([[1, 2, 3], [1, 2, 0]], dtype=torch.int32)
        mask = backbone._build_mask(positions, 0)
        assert mask.shape == (2, 4, 4)
        # диагональ всегда False (токен видит сам себя)
        diag = torch.arange(4)
        assert not mask[:, diag, diag].any()


def test_save_load_roundtrip(tmp_path) -> None:
    backbone = _backbone().eval()
    embedders = _embedders()
    events = _events()
    with torch.no_grad():
        expected = backbone(embedders, *events)
    backbone.save(tmp_path)
    loaded = Backbone.load(tmp_path).eval()
    with torch.no_grad():
        actual = loaded(embedders, *events)
    assert torch.allclose(actual, expected, atol=1e-5)
