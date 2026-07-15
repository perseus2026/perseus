import shutil
import types
from functools import cached_property
from pathlib import Path

import polars as pl

from perseus.core.event_store import base


class EventsWriter(base.EventsWriter):
    def __init__(self, path: Path, /) -> None:
        super().__init__(path)

        self._data: pl.DataFrame | None = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> bool | None:
        if exc_type is None:
            self._data.write_parquet(self.path / "data.pq")
            self._index.write_parquet(self.path / "index.pq")
        elif self.path.exists():
            shutil.rmtree(self.path)

        return super().__exit__(exc_type, exc_value, traceback)

    def _write(self, events: pl.DataFrame, /) -> None:
        if self._data is None:
            self._data = pl.DataFrame()
            self._index = pl.DataFrame(
                schema={
                    "client_id": pl.String(),
                    "from_index": pl.UInt32(),
                    "to_index": pl.UInt32(),
                },
            )

        self._index.vstack(
            events.lazy()
            .select("client_id")
            .with_row_index()
            .group_by("client_id")
            .agg(
                from_index=pl.col("index").min() + len(self._data),
                to_index=pl.col("index").max() + 1 + len(self._data),
            )
            .collect(),
            in_place=True,
        )
        self._data.vstack(events, in_place=True)


class EventsReader(base.EventsReader):
    def _read_events(self, client_id: str) -> pl.DataFrame:
        if (index := self._index.get(client_id)) is None:
            raise KeyError(f"unknown {client_id = }")
        from_index, to_index = index
        return self._data[from_index:to_index]

    @cached_property
    def _index(self) -> dict[str, tuple[int, int]]:
        return {
            client_id: (from_index, to_index)
            for client_id, from_index, to_index in pl.read_parquet(self.path / "index.pq").iter_rows()
        }

    @cached_property
    def _data(self) -> pl.DataFrame:
        return pl.read_parquet(self.path / "data.pq")
