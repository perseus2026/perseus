"""Checkpoint tests: Prepared and Trained save/load round-trip."""

import polars as pl

from perseus.core import checkpoint
from perseus.core.backbone import Backbone
from perseus.core.config import Config
from perseus.core.encoders.id import Embedder as IdEmbedder
from perseus.core.encoders.id import Preprocessor as IdPreprocessor
from perseus.core.tasks.classification.target import Artifacts as ClsArtifacts
from perseus.core.tasks.classification.target import Head as ClsHead
from perseus.core.tasks.classification.target import Layer as ClsLayer
from perseus.core.tasks.classification.target import Preprocessor as ClsPreprocessor
from tests.conftest import MINIMAL_CONFIG

DIM = 8


def _id_preprocessor() -> IdPreprocessor:
    return IdPreprocessor(pl.DataFrame({"value": ["x"], "index": [1]}), has_unk=False)


def test_prepared_save_load_roundtrip(tmp_path) -> None:
    config = Config.model_validate(MINIMAL_CONFIG)
    prepared = checkpoint.Prepared(config, {"event": _id_preprocessor()})
    prepared.save(tmp_path / "ckpt")
    loaded = checkpoint.Prepared.load(tmp_path / "ckpt")
    assert loaded.config.task.type == "classification"
    assert "event" in loaded.encoder_to_preprocessor
    assert loaded.encoder_to_preprocessor["event"].values["value"].to_list() == ["x"]


def test_trained_save_load_roundtrip(tmp_path) -> None:
    config = Config.model_validate(MINIMAL_CONFIG)
    cls_pre = ClsPreprocessor.fit(pl.Series(["a", "b"]), ClsArtifacts(), {})
    backbone = Backbone(
        DIM,
        history_aggregator_config={"type": "bert", "max_events_per_sequence": 16, "params": {"dropout": 0.0}},
        event_aggregator_config={"type": "sum", "num_features": 1, "params": {}},
    )
    head = ClsHead.init(cls_pre, ClsLayer.init(cls_pre, DIM))

    trained = checkpoint.Trained(
        config,
        {"event": _id_preprocessor()},
        {"event": IdEmbedder(2, DIM)},
        backbone,
        head,
    )
    trained.save(tmp_path / "ckpt")
    loaded = checkpoint.Trained.load(tmp_path / "ckpt")

    assert loaded.config.task.type == "classification"
    assert isinstance(loaded.backbone, Backbone)
    assert loaded.head.labels == ["a", "b"]
    assert "event" in loaded.encoder_to_embedder
