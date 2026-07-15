"""Dataset.save/load round-trip test (on a classification task)."""

import datetime as dt

import polars as pl

from perseus.core.config import Config
from perseus.core.dataset import Dataset
from perseus.core.event_store import EventStore
from perseus.core.tasks.classification.target import Artifacts, Preprocessor
from tests.conftest import MINIMAL_CONFIG


def test_dataset_save_load_roundtrip(tmp_path) -> None:
    config = Config.model_validate(MINIMAL_CONFIG)

    event_store = EventStore(tmp_path / "events_src", backend="disk")
    with event_store.open_writer() as writer:
        writer.write(
            pl.DataFrame(
                {
                    "client_id": ["c1", "c2"],
                    "timestamp": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 1)],
                    "position": [1, 1],
                    "event": [1, 2],
                },
            ),
        )

    preprocessor = Preprocessor.fit(pl.Series(["a", "b"]), Artifacts(), {})
    dataset = Dataset(
        train_samples=pl.DataFrame({"client_id": ["c1"], "target": ["a"]}),
        train_artifacts=Artifacts(),
        test_samples=pl.DataFrame({"client_id": ["c2"], "target": ["b"]}),
        test_artifacts=Artifacts(),
        event_store=event_store,
        task_preprocessor=preprocessor,
    )

    dataset.save(tmp_path / "ds")
    loaded = Dataset.load(tmp_path / "ds", config)

    assert loaded.train_samples["client_id"].to_list() == ["c1"]
    assert loaded.test_samples["client_id"].to_list() == ["c2"]
    assert loaded.task_preprocessor.label_to_index == {"a": 0, "b": 1}
    # event store has been relocated and is readable
    assert loaded.event_store.open_reader(512).read("c1")["event"].to_list() == [1]
