"""Tests for the disk backend of the event store (Arrow file)."""

import datetime as dt

import polars as pl
import pytest

from perseus.core.event_store import EventStore


def _events() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "client_id": ["c1", "c1", "c2"],
            "timestamp": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2), dt.datetime(2024, 1, 1)],
            "position": [1, 2, 1],
            "event": [10, 20, 30],
        },
    )


@pytest.fixture
def store(tmp_path) -> EventStore:
    store = EventStore(tmp_path / "store", backend="disk")
    with store.open_writer() as writer:
        writer.write(_events())
    return store


def test_writer_creates_arrow_and_index(store: EventStore) -> None:
    assert (store.path / "data.arrow").exists()
    assert (store.path / "index.pq").exists()


def test_reader_returns_client_events(store: EventStore) -> None:
    reader = store.open_reader(512)
    assert sorted(reader.read("c1")["event"].to_list()) == [10, 20]
    assert reader.read("c2")["event"].to_list() == [30]


def test_reader_unknown_client_raises(store: EventStore) -> None:
    reader = store.open_reader(512)
    with pytest.raises(KeyError, match="unknown"):
        reader.read("nope")


def test_max_events_per_sequence_truncates(store: EventStore) -> None:
    reader = store.open_reader(1)
    assert reader.read("c1")["position"].to_list() == [1]
