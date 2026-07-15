import logging

import polars as pl
from tqdm import tqdm

from perseus import core

logger = logging.getLogger(__name__)


def add_events(events: pl.DataFrame, name: str, source: str) -> None:
    source_to_path = core.event_hub.find_source_to_path()
    if (path := source_to_path.get(source)) is None:
        raise ValueError(f"unknown {source = }")

    if events.schema.get("client_id") != pl.String() or events["client_id"].is_null().any():
        raise ValueError("events must have non-empty column `client_id` with dtype `pl.String()`")
    if events.schema.get("timestamp") != pl.Datetime("ns") or events["timestamp"].is_null().any():
        raise ValueError("events must have non-empty column `timestamp` with dtype `pl.Datetime('ns')`")

    events = (
        core.event_hub.distribute_to_partitions(events.lazy())
        .sort("timestamp", "client_id")
        .with_columns(date=pl.col("timestamp").dt.date())
        .collect()
    )

    for (date,), date_events in tqdm(events.group_by("date", maintain_order=True), desc=f"adding {name} events"):
        (date_dir := path / name / str(date)).mkdir(parents=True, exist_ok=True)
        for (partition_id,), partition_events in date_events.group_by("partition_id", maintain_order=True):
            partition_events.drop("partition_id").write_parquet(date_dir / core.event_hub.filename(partition_id))

    logger.info(
        "saved %d %s events from %s to %s for %d clients",
        len(events),
        name,
        events["timestamp"].dt.date().min(),
        events["timestamp"].dt.date().max(),
        events["client_id"].n_unique(),
    )
