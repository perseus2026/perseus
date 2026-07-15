"""Тесты RAM-бэкенда event store и диспетчера EventStore."""

import datetime as dt

import polars as pl
import pytest

from perseus.core.event_store import EventStore


def _events() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "client_id": ["c1", "c1", "c1", "c2"],
            "timestamp": [
                dt.datetime(2024, 1, 1),
                dt.datetime(2024, 1, 2),
                dt.datetime(2024, 1, 3),
                dt.datetime(2024, 1, 1),
            ],
            "position": [1, 2, 3, 1],
            "event": ["a", "b", "c", "d"],
        },
    )


@pytest.fixture
def store(tmp_path) -> EventStore:
    path = tmp_path / "store"
    path.mkdir()
    store = EventStore(path, backend="ram")
    with store.open_writer() as writer:
        writer.write(_events())
    return store


def test_reader_returns_client_events(store: EventStore) -> None:
    reader = store.open_reader(512)
    assert reader.read("c1")["event"].to_list() == ["a", "b", "c"]
    assert reader.read("c2")["event"].to_list() == ["d"]


def test_reader_unknown_client_raises(store: EventStore) -> None:
    reader = store.open_reader(512)
    with pytest.raises(KeyError, match="unknown"):
        reader.read("does-not-exist")


def test_max_events_per_sequence_truncates(store: EventStore) -> None:
    reader = store.open_reader(2)
    assert reader.read("c1")["position"].to_list() == [1, 2]


def test_before_filter_and_position_reindex(store: EventStore) -> None:
    reader = store.open_reader(512)
    out = reader.read("c1", before=dt.datetime(2024, 1, 3))
    # остаются события 2024-01-01 и 2024-01-02; позиция переиндексируется с 1
    assert out["event"].to_list() == ["a", "b"]
    assert out["position"].to_list() == [1, 2]


def test_writer_removes_path_on_exception(tmp_path) -> None:
    store = EventStore(tmp_path / "store", backend="ram")
    with pytest.raises(RuntimeError, match="boom"), store.open_writer() as writer:  # noqa: PT012
        writer.write(_events())
        raise RuntimeError("boom")
    assert not (tmp_path / "store").exists()


def test_default_path_is_tempdir() -> None:
    store = EventStore(backend="ram")
    assert store.path.exists()


def test_move_relocates_store(store: EventStore, tmp_path) -> None:
    dst = tmp_path / "moved"
    store.move(dst)
    assert store.path == dst
    reader = store.open_reader(512)
    assert reader.read("c2")["event"].to_list() == ["d"]


def test_unknown_backend_raises(tmp_path) -> None:
    store = EventStore(tmp_path, backend="ram")
    store.backend = "bogus"
    with pytest.raises(RuntimeError, match="unknown backend"):
        store.open_writer()
    with pytest.raises(RuntimeError, match="unknown backend"):
        store.open_reader(512)
