"""Пошаговые тесты api.make_backbone_embeddings."""

import datetime as dt

import polars as pl

from perseus.api.make_backbone_embeddings import (
    _create_dataloader,
    _filter_samples,
    _make_embeddings,
    _prepare_samples,
    _static_transform_samples,
)
from tests.unit.api.conftest import DIM, make_samples


def test_prepare_samples_adds_timestamp_when_missing() -> None:
    samples = pl.DataFrame({"_index": [0], "client_id": ["c1"]})
    out = _prepare_samples(samples)
    assert "timestamp" in out.schema
    assert set(out.columns) == {"_index", "client_id", "timestamp"}


def test_prepare_samples_keeps_existing_timestamp_and_context() -> None:
    samples = pl.DataFrame(
        {
            "_index": [0],
            "client_id": ["c1"],
            "timestamp": [dt.datetime(2024, 1, 1)],
            "context": [{"city": 1}],
        },
    )
    out = _prepare_samples(samples)
    assert "context" in out.schema
    assert out["timestamp"].to_list() == [dt.datetime(2024, 1, 1)]


def test_filter_samples_drops_samples_before_first_event() -> None:
    samples = pl.DataFrame(
        {
            "client_id": ["c1", "c2"],
            "timestamp": [dt.datetime(2024, 2, 1), dt.datetime(2023, 1, 1)],
        },
    )
    events = pl.DataFrame(
        {
            "client_id": ["c1", "c2"],
            "timestamp": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 1)],
        },
    )
    out = _filter_samples(samples, events)
    # c2 имеет sample (2023) раньше первого события (2024) → отфильтрован
    assert out["client_id"].to_list() == ["c1"]


def test_static_transform_samples_noop_without_context(trained_checkpoint) -> None:
    samples = make_samples()
    assert _static_transform_samples(samples, trained_checkpoint).equals(samples)


def test_create_dataloader_and_make_embeddings(trained_checkpoint, event_store_with_events) -> None:
    reader = event_store_with_events.open_reader(512)
    dataloader = _create_dataloader(make_samples(), reader, trained_checkpoint)
    embeddings = _make_embeddings(dataloader, trained_checkpoint)

    assert set(embeddings.columns) == {"_index", "embeddings"}
    assert embeddings.height == 3
    assert embeddings.schema["embeddings"] == pl.Array(pl.Float32, DIM)
    assert sorted(embeddings["_index"].to_list()) == [0, 1, 2]
