import logging

import polars as pl

from perseus import core

logger = logging.getLogger(__name__)


def list_events(source: str) -> None:
    source_to_path = core.event_hub.find_source_to_path()
    if (path := source_to_path.get(source)) is None:
        raise ValueError(f"unknown {source = }")

    for events_path in sorted(path.iterdir()):
        events_dates_path = sorted(events_path.iterdir())
        events_schema = pl.scan_parquet(events_path / "*" / "*").collect_schema()
        logger.info(
            "%s events available for %d days from %s to %s with following attributes: %s",
            events_path.name,
            len(events_dates_path),
            events_dates_path[0].name,
            events_dates_path[-1].name,
            ", ".join(col for col in events_schema if col not in ("timestamp", "client_id")),
        )
