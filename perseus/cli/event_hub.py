import datetime as dt
from pathlib import Path

import polars as pl
from cyclopts import App

from perseus import api

app = App()


@app.command
def add_events(path: Path, /, *, name: str, source: str = "event_hub") -> None:
    api.add_events(pl.read_parquet(path), name, source)


@app.command
def list_events(*, source: str = "event_hub") -> None:
    api.list_events(source)


@app.command
def delete_events(
    *,
    name: str,
    date_from: dt.date | None = None,
    date_to: dt.date | None = None,
    source: str = "event_hub",
) -> None:
    api.delete_events(name, source, date_from, date_to)
