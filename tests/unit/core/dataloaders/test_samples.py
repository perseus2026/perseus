"""Тесты dataloaders.samples: _pad_sequences, _collate_samples, __getitem__, _init_worker."""

import datetime as dt
import os

import polars as pl
import torch

from perseus.core.dataloaders.samples import (
    _collate_samples,
    _init_worker_samples,
    _pad_sequences,
    _Sample,
    _SamplesDataset,
)
from perseus.core.encoders.id import Preprocessor as IdPreprocessor
from perseus.core.event_store import EventStore
from perseus.core.tasks.classification.target import Artifacts as ClsArtifacts
from perseus.core.tasks.classification.target import Preprocessor as ClsPreprocessor


class TestPadSequences:
    def test_left_pad_1d(self) -> None:
        out = _pad_sequences([torch.tensor([1, 2, 3]), torch.tensor([4, 5])], side="left")
        assert out.tolist() == [[1, 2, 3], [0, 4, 5]]

    def test_right_pad_1d(self) -> None:
        out = _pad_sequences([torch.tensor([1, 2, 3]), torch.tensor([4, 5])], side="right")
        assert out.tolist() == [[1, 2, 3], [4, 5, 0]]

    def test_pad_2d(self) -> None:
        out = _pad_sequences([torch.ones(3, 2), torch.ones(1, 2)], side="left")
        assert out.shape == (2, 3, 2)
        # первая строка второго элемента дополнена нулями слева
        assert out[1, 0].tolist() == [0.0, 0.0]


def _sample(positions: list[int], *, context=None, extras=None, target=None) -> _Sample:
    n = len(positions)
    return _Sample(
        events_features={"event": torch.tensor(positions, dtype=torch.int32)},
        events_positions=torch.tensor(positions, dtype=torch.int32),
        events_timestamps=torch.arange(n, dtype=torch.float32),
        sample_timestamp=float(n),
        context_features=context,
        extras=extras,
        target=target,
    )


class TestCollateSamples:
    def test_basic_padding_and_stacking(self) -> None:
        out = _collate_samples([_sample([1, 2, 3]), _sample([1, 2])])
        assert out.events_features["event"].shape == (2, 3)
        assert out.events_positions.shape == (2, 3)
        assert out.sample_timestamp.tolist() == [3.0, 2.0]
        assert out.context_features is None
        assert out.extras is None

    def test_with_context_and_extras(self) -> None:
        out = _collate_samples(
            [
                _sample([1, 2], context={"city": torch.randn(4)}, extras={"_index": 0}),
                _sample([1], context={"city": torch.randn(4)}, extras={"_index": 1}),
            ],
        )
        assert out.context_features["city"].shape == (2, 4)
        assert out.extras == {"_index": [0, 1]}

    def test_with_task_preprocessor_target(self) -> None:
        pre = ClsPreprocessor.fit(pl.Series(["a", "b"]), ClsArtifacts(), {})
        out = _collate_samples([_sample([1], target=0), _sample([1], target=1)], task_preprocessor=pre)
        assert out.target.tolist() == [0, 1]
        assert out.target.dtype == torch.long


def test_init_worker_sets_polars_threads() -> None:
    _init_worker_samples(0)
    assert os.environ["POLARS_MAX_THREADS"] == "1"


def test_dataset_getitem(tmp_path) -> None:
    path = tmp_path / "store"
    path.mkdir()
    store = EventStore(path, backend="ram")
    with store.open_writer() as writer:
        writer.write(
            pl.DataFrame(
                {
                    "client_id": ["c1", "c1", "c2"],
                    "timestamp": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2), dt.datetime(2024, 1, 1)],
                    "position": [1, 2, 1],
                    "event": [3, 4, 5],
                },
            ),
        )
    reader = store.open_reader(512)
    id_pre = IdPreprocessor(pl.DataFrame({"value": ["x"], "index": [1]}))
    dataset = _SamplesDataset(
        pl.DataFrame({"client_id": ["c1"], "timestamp": [dt.datetime(2024, 1, 3)]}),
        reader,
        {"event": id_pre},
        {},
    )
    sample = dataset[0]
    assert sample.events_features["event"].dtype == torch.int32
    assert sample.events_features["event"].tolist() == [3, 4]
    assert sample.events_positions.tolist() == [1, 2]
    assert sample.context_features is None
    assert len(dataset) == 1
