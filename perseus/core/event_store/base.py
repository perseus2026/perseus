import datetime as dt
import logging
import typing as t
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from pathlib import Path

import polars as pl

from perseus import utils

logger = logging.getLogger(__name__)


class EventsWriter(AbstractContextManager):
    def __init__(self, path: Path, /) -> None:
        self.path = path

    def __enter__(self) -> t.Self:
        return self

    @utils.timer(logger, "writing events to event store")
    def write(self, events: pl.DataFrame, /) -> None:
        return self._write(events)

    @abstractmethod
    def _write(self, events: pl.DataFrame, /) -> None:
        raise NotImplementedError


class EventsReader(ABC):
    def __init__(self, path: Path, /, *, max_events_per_sequence: int) -> None:
        self.path = path
        self.max_events_per_sequence = max_events_per_sequence

    def read(self, client_id: str, /, *, before: dt.date | dt.datetime | None = None) -> pl.DataFrame:
        events = self._read_events(client_id)
        if before is not None:
            events = (
                events.lazy()
                .filter(pl.col("timestamp") < before)
                .with_columns(position=pl.col("position") - pl.col("position").min() + 1)
                .collect()
            )

        return events.filter(pl.col("position") <= self.max_events_per_sequence)

    @abstractmethod
    def _read_events(self, client_id: str) -> pl.DataFrame:
        raise NotImplementedError
