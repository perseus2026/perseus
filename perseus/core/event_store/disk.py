import io
import shutil
import types
from functools import cached_property
from pathlib import Path

import polars as pl
import pyarrow as pa

from perseus.core.event_store import base


class EventsWriter(base.EventsWriter):
    COMPRESSION = "zstd"

    def __init__(self, path: Path, /) -> None:
        super().__init__(path)

        self._schema: pa.Schema | None = None
        self._writer: pa.ipc.RecordBatchFileWriter | None = None
        self._index: pl.Series | None = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> bool | None:
        if self._writer is not None:
            self._writer.close()

        if exc_type is None:
            self._index.to_frame("client_id").write_parquet(self.path / "index.pq")
        elif self.path.exists():
            shutil.rmtree(self.path)

        return super().__exit__(exc_type, exc_value, traceback)

    def _write(self, events: pl.DataFrame, /) -> None:
        if self._schema is None:
            self._schema = events[0].to_arrow().schema
            self.path.mkdir(parents=True, exist_ok=True)
            self._writer = pa.ipc.new_file(
                self.path / "data.arrow",
                self._schema,
                options=pa.ipc.IpcWriteOptions(compression=pa.Codec(self.COMPRESSION)),
            )
            self._index = pl.Series(dtype=pl.String())

        index = []
        for (client_id,), client_events in events.group_by("client_id"):
            client_events_copy = pl.DataFrame.deserialize(io.BytesIO(client_events.serialize()))
            self._writer.write_table(client_events_copy.to_arrow().combine_chunks())
            index.append(client_id)
        self._index = self._index.extend(pl.Series(index))


class EventsReader(base.EventsReader):
    def _read_events(self, client_id: str) -> pl.DataFrame:
        if (index := self._index.get(client_id)) is None:
            raise KeyError(f"unknown {client_id = }")
        return pl.from_arrow(self._data.get_record_batch(index))

    @cached_property
    def _index(self) -> dict[str, int]:
        return dict(
            pl.scan_parquet(self.path / "index.pq", row_index_name="index")
            .select("client_id", "index")
            .collect()
            .iter_rows(),
        )

    @cached_property
    def _data(self) -> pa.ipc.RecordBatchFileReader:
        return pa.ipc.open_file(self.path / "data.arrow")
