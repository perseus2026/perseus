import shutil
import tempfile
import typing as t
import weakref
from pathlib import Path

from perseus.core.event_store import disk, ram
from perseus.core.event_store.base import EventsReader, EventsWriter

__all__ = [
    "EventStore",
    "EventsReader",
    "EventsWriter",
]


class EventStore:
    def __init__(self, path: Path | None = None, /, *, backend: t.Literal["disk", "ram"]) -> None:
        self.backend = backend

        self._finalizer: weakref.finalize | None
        if path is None:
            self.path = Path(tempfile.mkdtemp())
            self._finalizer = weakref.finalize(self, shutil.rmtree, self.path, ignore_errors=True)
        else:
            self.path = path
            self._finalizer = None

    def open_writer(self) -> EventsWriter:
        match self.backend:
            case "disk":
                writer_cls = disk.EventsWriter
            case "ram":
                writer_cls = ram.EventsWriter
            case _:
                raise RuntimeError(f"unknown backend = {self.backend}")
        return writer_cls(self.path)

    def open_reader(self, max_events_per_sequence: int, /) -> EventsReader:
        match self.backend:
            case "disk":
                reader_cls = disk.EventsReader
            case "ram":
                reader_cls = ram.EventsReader
            case _:
                raise RuntimeError(f"unknown backend = {self.backend}")
        return reader_cls(self.path, max_events_per_sequence=max_events_per_sequence)

    def move(self, path: Path, /) -> None:
        path.mkdir(parents=True, exist_ok=True)

        for src_path in sorted(self.path.iterdir()):
            dst_path = path / src_path.name
            if dst_path.is_file():
                dst_path.unlink()
            elif dst_path.is_dir():
                shutil.rmtree(dst_path)
            shutil.move(src_path, path)
        shutil.rmtree(self.path)
        self.path = path

        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None
