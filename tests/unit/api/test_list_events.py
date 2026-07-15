"""Tests for api.list_events: listing the events available in a source."""

import datetime as dt
import logging

import polars as pl
import pytest

from perseus.api.add_events import add_events
from perseus.api.list_events import list_events


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERNAL_STORAGE_EVENT_HUB", str(tmp_path))
    return tmp_path


def _seed(name: str) -> None:
    events = pl.DataFrame(
        {
            "client_id": ["c1", "c2"],
            "timestamp": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 3)],
            "amount": [1, 2],
        },
        schema_overrides={"timestamp": pl.Datetime("ns")},
    )
    add_events(events, name, source="event_hub")


def test_unknown_source_raises(storage) -> None:
    with pytest.raises(ValueError, match="unknown source"):
        list_events(source="missing")


def test_logs_summary_per_event(storage, caplog) -> None:
    _seed("purchase")
    _seed("view")

    with caplog.at_level(logging.INFO):
        list_events(source="event_hub")

    text = caplog.text
    # one line per event type, with the date range and attributes excluding service columns
    assert "purchase events available" in text
    assert "view events available" in text
    assert "2024-01-01" in text
    assert "2024-01-03" in text
    assert "amount" in text
    assert "client_id" not in text
    assert "timestamp" not in text
