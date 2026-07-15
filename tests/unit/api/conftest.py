"""Shared builders for step-by-step tests of the api layer (classification, event id encoder)."""

import datetime as dt

import polars as pl
import pytest

from perseus.core import checkpoint as ckpt
from perseus.core.backbone import Backbone
from perseus.core.config import Config
from perseus.core.dataset import Dataset
from perseus.core.encoders.id import Embedder as IdEmbedder
from perseus.core.encoders.id import Preprocessor as IdPreprocessor
from perseus.core.event_store import EventStore
from perseus.core.tasks.classification.target import Artifacts as ClsArtifacts
from perseus.core.tasks.classification.target import Head as ClsHead
from perseus.core.tasks.classification.target import Layer as ClsLayer
from perseus.core.tasks.classification.target import Preprocessor as ClsPreprocessor
from tests.conftest import MINIMAL_CONFIG

DIM = 8


def make_config() -> Config:
    return Config.model_validate(MINIMAL_CONFIG)


def make_id_preprocessor() -> IdPreprocessor:
    return IdPreprocessor(pl.DataFrame({"value": ["x"], "index": [1]}), has_unk=False)


def make_backbone() -> Backbone:
    return Backbone(
        DIM,
        history_aggregator_config={"type": "bert", "max_events_per_sequence": 512, "params": {"dropout": 0.0}},
        event_aggregator_config={"type": "sum", "num_features": 1, "params": {}},
    )


def make_task_preprocessor() -> ClsPreprocessor:
    return ClsPreprocessor.fit(pl.Series(["a", "b"]), ClsArtifacts(), {})


@pytest.fixture
def prepared_checkpoint() -> ckpt.Prepared:
    return ckpt.Prepared(make_config(), {"event": make_id_preprocessor()})


@pytest.fixture
def trained_checkpoint() -> ckpt.Trained:
    cls_pre = make_task_preprocessor()
    head = ClsHead.init(cls_pre, ClsLayer.init(cls_pre, DIM))
    return ckpt.Trained(
        make_config(),
        {"event": make_id_preprocessor()},
        {"event": IdEmbedder(10, DIM)},
        make_backbone().eval(),
        head,
    )


@pytest.fixture
def event_store_with_events(tmp_path) -> EventStore:
    path = tmp_path / "store"
    path.mkdir()
    store = EventStore(path, backend="ram")
    with store.open_writer() as writer:
        writer.write(
            pl.DataFrame(
                {
                    "client_id": ["c1", "c1", "c2", "c3"],
                    "timestamp": [
                        dt.datetime(2024, 1, 1),
                        dt.datetime(2024, 1, 2),
                        dt.datetime(2024, 1, 1),
                        dt.datetime(2024, 1, 1),
                    ],
                    "position": [1, 2, 1, 1],
                    "event": [1, 2, 3, 4],
                },
            ),
        )
    return store


def make_samples() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "_index": [0, 1, 2],
            "client_id": ["c1", "c2", "c3"],
            "timestamp": [dt.datetime(2024, 2, 1)] * 3,
        },
    )


@pytest.fixture
def dataset(event_store_with_events) -> Dataset:
    return Dataset(
        train_samples=pl.DataFrame(
            {"client_id": ["c1", "c2"], "timestamp": [dt.datetime(2024, 2, 1)] * 2, "target": [0, 1]},
        ),
        train_artifacts=ClsArtifacts(),
        # test target — a string label (as in the pipeline: the test target is not statically transformed),
        # so that the evaluator compares it against most_probable_label
        test_samples=pl.DataFrame(
            {"client_id": ["c3"], "timestamp": [dt.datetime(2024, 2, 1)], "target": ["a"]},
        ),
        test_artifacts=ClsArtifacts(),
        event_store=event_store_with_events,
        task_preprocessor=make_task_preprocessor(),
    )
