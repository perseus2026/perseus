"""Isolated smoke tests for the top-level api stages (a full run of each entrypoint separately)."""

import polars as pl

from perseus import core
from perseus.api.distribute_samples import distribute_samples
from perseus.api.embed_artifacts import embed_artifacts
from perseus.api.fit_model import fit_model
from perseus.api.make_backbone_embeddings import make_backbone_embeddings
from perseus.api.make_head_predictions import make_head_predictions
from perseus.api.prepare_dataset import prepare_dataset
from tests.integration.conftest import (
    SAMPLE_DATE,
    build_dataset,
    build_prepared,
    build_trained,
    partition_of,
    setup_purchase_source,
)
from tests.unit.api.conftest import DIM, make_config


def test_smoke_distribute_samples() -> None:
    samples = pl.DataFrame({"client_id": ["c1", "c2", "c3", "c1"], "value": [1, 2, 3, 4]})
    partitions = list(distribute_samples(samples))
    assert sum(len(part) for _, part in partitions) == 4
    assert {c for _, part in partitions for c in part["client_id"]} == {"c1", "c2", "c3"}


def test_smoke_prepare_dataset(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_STORAGE_EVENT_HUB", str(tmp_path / "hub"))
    setup_purchase_source(tmp_path / "hub", ["c1", "c2", "c3", "c4"])

    config = make_config()
    train = pl.DataFrame({"client_id": ["c1", "c2"], "timestamp": [SAMPLE_DATE] * 2, "target": ["a", "b"]})
    test = pl.DataFrame({"client_id": ["c3", "c4"], "timestamp": [SAMPLE_DATE] * 2, "target": ["a", "b"]})

    prepared, dataset = prepare_dataset(config, train, test)

    assert isinstance(prepared, core.checkpoint.Prepared)
    assert "event" in prepared.encoder_to_preprocessor
    assert dataset.task_preprocessor.label_to_index == {"a": 0, "b": 1}
    assert dataset.train_samples.height == 2
    # events are written to the event store and can be read back
    reader = dataset.event_store.open_reader(config.max_events_per_sequence)
    assert reader.read("c1").height >= 1


def test_smoke_fit_model(tmp_path) -> None:
    prepared = build_prepared()
    dataset = build_dataset(tmp_path)

    trained = fit_model(prepared, dataset)

    assert isinstance(trained, core.checkpoint.Trained)
    assert "event" in trained.encoder_to_embedder
    assert trained.head.labels == ["a", "b"]
    assert isinstance(trained.backbone, core.Backbone)


def test_smoke_make_backbone_embeddings(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_STORAGE_EVENT_HUB", str(tmp_path / "hub"))
    setup_purchase_source(tmp_path / "hub", ["c1"])
    trained = build_trained()

    pid = partition_of("c1")
    samples = pl.DataFrame({"_index": [0], "client_id": ["c1"], "timestamp": [SAMPLE_DATE]})

    embeddings = make_backbone_embeddings(pid, samples, trained)

    assert embeddings is not None
    assert embeddings.schema["embeddings"] == pl.Array(pl.Float32, DIM)
    assert embeddings.height == 1


def test_smoke_make_head_predictions(monkeypatch) -> None:
    trained = build_trained()
    samples = pl.DataFrame({"_index": [0, 1], "client_id": ["c1", "c2"]})
    embeddings = pl.DataFrame(
        {"_index": [0, 1], "embeddings": [[0.1] * DIM, [0.2] * DIM]},
        schema_overrides={"embeddings": pl.Array(pl.Float32, DIM)},
    )

    predictions = make_head_predictions(samples, embeddings, trained)

    assert predictions.height == 2
    assert predictions["prediction"].dtype == pl.Struct({"a": pl.Float32, "b": pl.Float32})


def test_smoke_embed_artifacts() -> None:
    # a classification task has no artifact features -> embed returns None (a full entrypoint run)
    assert embed_artifacts(None, build_trained()) is None
