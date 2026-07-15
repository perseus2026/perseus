"""Tests for event_hub: distribution across partitions, file names, Reader."""

import datetime as dt

import polars as pl
import pytest

from perseus.core.config import Config
from perseus.core.event_hub import Reader, distribute_to_partitions, filename, partition_id
from tests.conftest import MINIMAL_CONFIG


def test_distribute_to_partitions_assigns_partition_id() -> None:
    samples = pl.DataFrame({"client_id": ["c1", "c2", "c3", "c1"]})
    out = distribute_to_partitions(samples)
    assert "partition_id" in out.schema
    assert len(out) == 4
    # partitions are in the range [1, 100]
    assert out["partition_id"].min() >= 1
    assert out["partition_id"].max() <= 100
    # a single client_id -> a single partition
    assert out.filter(pl.col("client_id") == "c1")["partition_id"].n_unique() == 1


def test_filename_and_partition_id_roundtrip() -> None:
    assert filename(1) == "0001.pq"
    assert filename(42) == "0042.pq"
    assert partition_id("0042.pq") == 42


@pytest.fixture
def config_fixture() -> Config:
    return Config.model_validate(MINIMAL_CONFIG)


def test_reader_init_missing_source_raises(config_fixture: Config, monkeypatch) -> None:
    # remove any internal_storage_* from the environment -> the event_hub source is not connected
    for key in list(__import__("os").environ):
        if key.lower().startswith("internal_storage_"):
            monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="need to connect sources"):
        Reader(config_fixture)


@pytest.fixture
def event_hub_dir(tmp_path, monkeypatch) -> "Path":  # noqa: F821
    # synthetic source: <storage>/purchase/<date>/0001.pq
    event_dir = tmp_path / "purchase" / "2024-01-01"
    event_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "client_id": ["c1", "c1", "c2"],
            "timestamp": [
                dt.datetime(2024, 1, 1, 10),
                dt.datetime(2024, 1, 1, 11),
                dt.datetime(2024, 1, 1, 12),
            ],
        },
    ).write_parquet(event_dir / "0001.pq")
    monkeypatch.setenv("INTERNAL_STORAGE_EVENT_HUB", str(tmp_path))
    return tmp_path


def test_reader_read_flattens_events(config_fixture: Config, event_hub_dir) -> None:
    reader = Reader(config_fixture)
    samples = pl.DataFrame(
        {
            "client_id": ["c1", "c2"],
            "_sample_timestamp": [dt.datetime(2024, 1, 1, 13), dt.datetime(2024, 1, 1, 12)],
        },
    )
    events = reader.read(1, samples)
    assert events is not None
    assert set(events.columns) >= {"client_id", "timestamp", "event", "position"}
    assert events["event"].unique().to_list() == ["purchase"]
    # c1 has 2 events, c2 has one
    assert events.filter(pl.col("client_id") == "c1").height == 2
    assert events.filter(pl.col("client_id") == "c2").height == 0


def test_reader_read_returns_none_for_empty_partition(config_fixture: Config, event_hub_dir) -> None:
    reader = Reader(config_fixture)
    # a partition with no events -> no 0099.pq file -> None
    samples = pl.DataFrame({"client_id": ["c1"]})
    assert reader.read(99, samples) is None


def test_readerconfig_fixture_cached_properties(config_fixture: Config, event_hub_dir) -> None:
    reader = Reader(config_fixture)
    # minimal config: only the "purchase" event with no attributes or multi-fields
    assert reader._event_to_multi_attributes == {}
    assert reader._attribute_to_multi == {"event": False}
    assert reader._event_to_max_duration == {}
    assert reader._event_to_max_count == {}


@pytest.fixture
def multi_attr_source(tmp_path, monkeypatch) -> Config:
    """Source with a multi-attribute purchase event and a scalar view, plus a config for them."""
    purchase_dir = tmp_path / "purchase" / "2024-01-28"
    purchase_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "client_id": ["c1", "c1"],
            "timestamp": [dt.datetime(2024, 1, 28, 10), dt.datetime(2024, 1, 28, 11)],
            "item": [[1, 2, 3, 4], [5, 6]],
        },
    ).write_parquet(purchase_dir / "0001.pq")

    view_dir = tmp_path / "view" / "2024-01-10"
    view_dir.mkdir(parents=True)
    pl.DataFrame(
        {"client_id": ["c1"], "timestamp": [dt.datetime(2024, 1, 10)], "page": [7]},
    ).write_parquet(view_dir / "0001.pq")

    monkeypatch.setenv("INTERNAL_STORAGE_EVENT_HUB", str(tmp_path))

    return Config.model_validate(
        {
            "task": {"type": "classification", "metrics": {"accuracy": None}},
            "events": {
                "purchase": {
                    "attributes": {"item": {"multi": True}},
                    "max_tokens": 3,
                    "priority": 1,
                    "max_events_per_sequence": 5,
                    "max_duration_per_sequence": "7d",
                },
                "view": {"attributes": {"page": None}, "priority": 0},
            },
            "max_duration_per_sequence": "30d",
            "features": {
                "item": {"located_in": {"event": True}, "encoder": {"type": "id"}},
                "page": {"located_in": {"event": True}, "encoder": {"type": "id"}},
            },
            "backbone": {"dim": 8},
        },
    )


def test_reader_config_properties_with_multi_attrs(multi_attr_source: Config) -> None:
    reader = Reader(multi_attr_source)
    assert reader._event_to_multi_attributes == {"purchase": ["item"]}
    assert reader._attribute_to_multi == {"event": False, "item": True, "page": False}
    assert reader._event_to_max_duration == {"purchase": "7d"}
    assert reader._event_to_max_count == {"purchase": 5}


def test_reader_read_multi_attrs_explodes_and_filters(multi_attr_source: Config) -> None:
    reader = Reader(multi_attr_source)
    samples = pl.DataFrame(
        {"client_id": ["c1"], "_sample_timestamp": [dt.datetime(2024, 2, 1)]},
    )
    events = reader.read(1, samples)
    assert events is not None
    assert set(events["event"].unique().to_list()) == {"purchase", "view"}
    assert "position" in events.columns
    # purchase events have item (after explode), view has null item
    purchase = events.filter(pl.col("event") == "purchase")
    view = events.filter(pl.col("event") == "view")
    assert purchase["item"].null_count() == 0
    assert view["item"].null_count() == view.height
    # max_tokens=3 -> each purchase event has at most 3 tokens (after explode, <= 3 in total per event)
    assert purchase.height <= 2 * 3  # 2 purchase events * 3 tokens
