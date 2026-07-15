"""Tests for api.delete_events: full deletion and deletion by date range."""

import datetime as dt
import logging

import polars as pl
import pytest

from perseus.api.add_events import add_events
from perseus.api.delete_events import delete_events


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERNAL_STORAGE_EVENT_HUB", str(tmp_path))
    return tmp_path


def _seed() -> None:
    """purchase events over three days: 01-01, 01-02, 01-03."""
    events = pl.DataFrame(
        {
            "client_id": ["c1", "c1", "c2"],
            "timestamp": [
                dt.datetime(2024, 1, 1),
                dt.datetime(2024, 1, 2),
                dt.datetime(2024, 1, 3),
            ],
        },
        schema_overrides={"timestamp": pl.Datetime("ns")},
    )
    add_events(events, "purchase", source="event_hub")


def _dates(storage) -> list[str]:
    return sorted(p.name for p in (storage / "purchase").iterdir())


def test_unknown_source_raises(storage) -> None:
    with pytest.raises(ValueError, match="unknown source"):
        delete_events("purchase", source="missing")


def test_missing_events_warns_and_returns(storage, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        delete_events("purchase", source="event_hub")
    assert "do not exist" in caplog.text


def test_delete_all(storage) -> None:
    _seed()
    delete_events("purchase", source="event_hub")
    assert not (storage / "purchase").exists()


def test_delete_from(storage) -> None:
    _seed()
    delete_events("purchase", source="event_hub", date_from=dt.date(2024, 1, 2))
    # dates >= 01-02 are deleted, only 01-01 remains
    assert _dates(storage) == ["2024-01-01"]


def test_delete_to(storage) -> None:
    _seed()
    delete_events("purchase", source="event_hub", date_to=dt.date(2024, 1, 2))
    # dates <= 01-02 are deleted, only 01-03 remains
    assert _dates(storage) == ["2024-01-03"]


def test_delete_range(storage) -> None:
    _seed()
    delete_events(
        "purchase",
        source="event_hub",
        date_from=dt.date(2024, 1, 2),
        date_to=dt.date(2024, 1, 2),
    )
    # only the middle of the range is deleted
    assert _dates(storage) == ["2024-01-01", "2024-01-03"]
