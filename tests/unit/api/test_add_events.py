"""Тесты api.add_events: валидация схемы и запись событий в партиции event-hub."""

import datetime as dt

import polars as pl
import pytest

from perseus.api.add_events import add_events
from perseus.core.event_hub import filename, partition_id


@pytest.fixture
def storage(tmp_path, monkeypatch):
    """Подключаем источник event_hub к временной директории."""
    monkeypatch.setenv("INTERNAL_STORAGE_EVENT_HUB", str(tmp_path))
    return tmp_path


def _events() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "client_id": ["c1", "c1", "c2"],
            "timestamp": [
                dt.datetime(2024, 1, 1, 10),
                dt.datetime(2024, 1, 2, 11),
                dt.datetime(2024, 1, 1, 12),
            ],
            "amount": [10, 20, 30],
        },
        schema_overrides={"timestamp": pl.Datetime("ns")},
    )


def test_unknown_source_raises(storage) -> None:
    with pytest.raises(ValueError, match="unknown source"):
        add_events(_events(), "purchase", source="missing")


def test_missing_client_id_raises(storage) -> None:
    events = _events().drop("client_id")
    with pytest.raises(ValueError, match="client_id"):
        add_events(events, "purchase", source="event_hub")


def test_null_client_id_raises(storage) -> None:
    events = _events().with_columns(client_id=pl.lit(None, dtype=pl.String()))
    with pytest.raises(ValueError, match="client_id"):
        add_events(events, "purchase", source="event_hub")


def test_wrong_timestamp_dtype_raises(storage) -> None:
    events = _events().with_columns(timestamp=pl.col("timestamp").dt.date())
    with pytest.raises(ValueError, match="timestamp"):
        add_events(events, "purchase", source="event_hub")


def test_writes_partitioned_by_date(storage) -> None:
    add_events(_events(), "purchase", source="event_hub")

    event_dir = storage / "purchase"
    dates = sorted(p.name for p in event_dir.iterdir())
    # события разнесены по двум датам
    assert dates == ["2024-01-01", "2024-01-02"]

    # все записанные файлы имеют валидное имя партиции и не содержат partition_id
    for date_dir in event_dir.iterdir():
        for pq in date_dir.iterdir():
            assert pq.name == filename(partition_id(pq.name))
            written = pl.read_parquet(pq)
            assert "partition_id" not in written.schema
            assert {"client_id", "timestamp", "amount"} <= set(written.columns)


def test_roundtrip_preserves_all_rows(storage) -> None:
    add_events(_events(), "purchase", source="event_hub")
    restored = pl.read_parquet(storage / "purchase" / "*" / "*")
    assert restored.height == 3
    assert sorted(restored["client_id"].to_list()) == ["c1", "c1", "c2"]
    assert sorted(restored["amount"].to_list()) == [10, 20, 30]
