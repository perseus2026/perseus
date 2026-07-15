"""Пошаговые тесты api.fit_model (init/create-шаги; полный цикл обучения не запускается)."""

import logging

import pytest
from accelerate import Accelerator, DataLoaderConfiguration

from perseus.api.fit_model import (
    _create_dataloaders,
    _create_early_stopping,
    _create_optimizer,
    _create_scheduler,
    _dynamic_transform_artifacts,
    _init_backbone,
    _init_embedders,
    _init_task_entities,
    _load_model,
    _log_parameters,
    _save_model,
    _test_epoch,
    _train_epoch,
    _unwrap,
)
from perseus.core import checkpoint as ckpt
from perseus.core import trackers
from perseus.core.backbone import Backbone
from perseus.core.config import Config
from tests.conftest import MINIMAL_CONFIG
from tests.unit.api.conftest import make_id_preprocessor


@pytest.fixture
def accelerator() -> Accelerator:
    return Accelerator(dataloader_config=DataLoaderConfiguration(non_blocking=True))


def test_init_embedders_wraps(accelerator, prepared_checkpoint) -> None:
    embedders = _init_embedders(accelerator, prepared_checkpoint)
    assert "event" in embedders
    assert "_wrapped" in embedders["event"].__dict__


def test_init_backbone(accelerator, prepared_checkpoint) -> None:
    backbone = _init_backbone(accelerator, prepared_checkpoint)
    assert isinstance(backbone, Backbone)
    assert "_wrapped" in backbone.__dict__


def test_init_task_entities(accelerator, prepared_checkpoint, dataset) -> None:
    _layer, _criterion, head, evaluator = _init_task_entities(accelerator, prepared_checkpoint, dataset)
    assert head.labels == ["a", "b"]
    assert "accuracy" in evaluator.name_to_metric


def test_create_optimizer_and_scheduler(accelerator, prepared_checkpoint, dataset) -> None:
    embedders = _init_embedders(accelerator, prepared_checkpoint)
    backbone = _init_backbone(accelerator, prepared_checkpoint)
    layer, criterion, _, _ = _init_task_entities(accelerator, prepared_checkpoint, dataset)
    optimizer = _create_optimizer(accelerator, embedders, backbone, layer, criterion, prepared_checkpoint)
    scheduler = _create_scheduler(accelerator, optimizer, prepared_checkpoint)
    assert optimizer is not None
    assert scheduler is not None
    _log_parameters(embedders, backbone, layer, criterion)
    _unwrap(accelerator, embedders, backbone, layer)
    assert "_wrapped" not in backbone.__dict__


def test_create_early_stopping_none(prepared_checkpoint) -> None:
    assert _create_early_stopping(prepared_checkpoint) is None


def test_create_early_stopping_configured() -> None:
    config = Config.model_validate(
        {**MINIMAL_CONFIG, "training": {"early_stopping": {"metric": "accuracy"}}},
    )
    prepared = ckpt.Prepared(config, {"event": make_id_preprocessor()})
    early_stopping = _create_early_stopping(prepared)
    assert early_stopping is not None
    assert early_stopping.metric == "accuracy"


def test_dynamic_transform_artifacts(prepared_checkpoint, dataset) -> None:
    train, test = _dynamic_transform_artifacts(prepared_checkpoint, dataset)
    # classification artifacts → None
    assert train is None
    assert test is None


def test_create_dataloaders(accelerator, prepared_checkpoint, dataset, caplog) -> None:
    with caplog.at_level(logging.INFO):
        train_dl, test_dl = _create_dataloaders(accelerator, prepared_checkpoint, dataset)
    train_batch = next(iter(train_dl))
    assert train_batch.target.tolist() == [0, 1] or train_batch.target.numel() >= 1
    test_batch = next(iter(test_dl))
    assert "_index" in test_batch.extras


def _full_setup(accelerator, prepared_checkpoint, dataset):
    embedders = _init_embedders(accelerator, prepared_checkpoint)
    backbone = _init_backbone(accelerator, prepared_checkpoint)
    layer, criterion, head, evaluator = _init_task_entities(accelerator, prepared_checkpoint, dataset)
    optimizer = _create_optimizer(accelerator, embedders, backbone, layer, criterion, prepared_checkpoint)
    scheduler = _create_scheduler(accelerator, optimizer, prepared_checkpoint)
    train_dl, test_dl = _create_dataloaders(accelerator, prepared_checkpoint, dataset)
    train_art, test_art = _dynamic_transform_artifacts(prepared_checkpoint, dataset)
    return (
        embedders,
        backbone,
        layer,
        criterion,
        head,
        evaluator,
        optimizer,
        scheduler,
        train_dl,
        test_dl,
        train_art,
        test_art,
    )


def test_train_test_epoch_and_save_load(accelerator, prepared_checkpoint, dataset, tmp_path) -> None:
    tracker = trackers.registry["logging"]()

    (
        embedders,
        backbone,
        layer,
        criterion,
        head,
        evaluator,
        optimizer,
        scheduler,
        train_dl,
        test_dl,
        train_art,
        test_art,
    ) = _full_setup(accelerator, prepared_checkpoint, dataset)

    total_step, _loss_buffer = _train_epoch(
        1,
        0,
        [],
        accelerator,
        embedders,
        backbone,
        layer,
        criterion,
        train_dl,
        train_art,
        optimizer,
        scheduler,
        prepared_checkpoint,
        dataset,
        tracker,
    )
    assert total_step >= 1

    _save_model(1, tmp_path, accelerator, embedders, backbone, layer)
    assert (tmp_path / "states" / "epoch=0001" / "backbone").exists()

    metrics = _test_epoch(
        1,
        tmp_path,
        accelerator,
        embedders,
        backbone,
        layer,
        head,
        evaluator,
        test_dl,
        test_art,
        prepared_checkpoint,
        dataset,
        tracker,
    )
    assert "accuracy" in metrics
    assert "overall" in metrics["accuracy"]

    _unwrap(accelerator, embedders, backbone, layer)
    _load_model(1, tmp_path, embedders, backbone, layer)  # не должно падать
