import typing as t
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from perseus.core import tasks
from perseus.core.config import Config
from perseus.core.event_store import EventStore


@dataclass
class Dataset:
    train_samples: pl.DataFrame
    train_artifacts: tasks.Artifacts
    test_samples: pl.DataFrame
    test_artifacts: tasks.Artifacts
    event_store: EventStore
    task_preprocessor: tasks.Preprocessor

    def save(self, path: Path, /) -> None:
        path.mkdir(parents=True, exist_ok=True)

        (train_path := path / "train").mkdir(parents=True, exist_ok=True)
        self.train_samples.write_parquet(train_path / "samples.pq")
        self.train_artifacts.save(train_path / "artifacts")

        (test_dir := path / "test").mkdir(parents=True, exist_ok=True)
        self.test_samples.write_parquet(test_dir / "samples.pq")
        self.test_artifacts.save(test_dir / "artifacts")

        self.event_store.move(path / "events")

        self.task_preprocessor.save(path / "preprocessor")

    @classmethod
    def load(cls, path: Path, config: Config, /) -> t.Self:
        artifacts_cls = tasks.registry[config.task.type].artifacts

        train_path = path / "train"
        train_samples = pl.read_parquet(train_path / "samples.pq")
        train_artifacts = artifacts_cls.load(train_path / "artifacts")

        test_path = path / "test"
        test_samples = pl.read_parquet(test_path / "samples.pq")
        test_artifacts = artifacts_cls.load(test_path / "artifacts")

        event_store = EventStore(path / "events", backend="disk")
        task_preprocessor = tasks.registry[config.task.type].preprocessor.load(path / "preprocessor")

        return cls(
            train_samples,
            train_artifacts,
            test_samples,
            test_artifacts,
            event_store,
            task_preprocessor,
        )
