import datetime as dt
import logging
import shutil

from perseus import core

logger = logging.getLogger(__name__)


def delete_events(
    name: str,
    source: str,
    date_from: dt.date | None = None,
    date_to: dt.date | None = None,
) -> None:
    source_to_path = core.event_hub.find_source_to_path()
    if (path := source_to_path.get(source)) is None:
        raise ValueError(f"unknown {source = }")

    events_path = path / name
    if not events_path.exists():
        logger.warning("%s events do not exist", name)
        return

    if date_from is None and date_to is None:
        shutil.rmtree(events_path)
        logger.info("deleted all %s events", name)
        return

    for date_events_path in sorted(events_path.iterdir()):
        date = dt.date.fromisoformat(date_events_path.name)
        if date_from and date < date_from:
            continue
        if date_to and date > date_to:
            continue
        shutil.rmtree(date_events_path)
        logger.info("deleted %s events for %s", name, date)
