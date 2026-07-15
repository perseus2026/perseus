import contextlib
import itertools as it
import logging
import shutil
import typing as t
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import torch
from accelerate import Accelerator, DataLoaderConfiguration, load_checkpoint_in_model
from accelerate.utils import broadcast_object_list
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from perseus import core

logger = logging.getLogger(__name__)


def fit_model(checkpoint: core.checkpoint.Prepared, dataset: core.Dataset, /) -> core.checkpoint.Trained:
    accelerator = Accelerator(dataloader_config=DataLoaderConfiguration(non_blocking=True))

    tracker = None
    if accelerator.is_main_process:
        tracker_config = checkpoint.config.training.tracker
        if tracker_config.type not in core.trackers.registry:
            raise ValueError(
                f"unknown tracker `{tracker_config.type}`; available: {sorted(core.trackers.registry)} "
                f"(some trackers require extra dependencies, e.g. `pip install perseus[clearml]`)",
            )
        tracker = core.trackers.registry[tracker_config.type](**tracker_config.params)

    encoder_to_embedder = _init_embedders(accelerator, checkpoint)
    backbone = _init_backbone(accelerator, checkpoint)
    task_layer, criterion, head, evaluator = _init_task_entities(accelerator, checkpoint, dataset)
    if accelerator.is_main_process:
        _log_parameters(encoder_to_embedder, backbone, task_layer, criterion)

    optimizer = _create_optimizer(accelerator, encoder_to_embedder, backbone, task_layer, criterion, checkpoint)
    scheduler = _create_scheduler(accelerator, optimizer, checkpoint)
    early_stopping = _create_early_stopping(checkpoint)

    train_dataloader, test_dataloader = _create_dataloaders(accelerator, checkpoint, dataset)
    train_transformed_artifacts, test_transformed_artifacts = _dynamic_transform_artifacts(checkpoint, dataset)

    with contextlib.ExitStack() as stack:
        cache_dir = Path(stack.enter_context(TemporaryDirectory())) if accelerator.is_main_process else None
        [cache_dir] = broadcast_object_list([cache_dir], from_process=0)

        total_step = 0
        loss_buffer = []
        for epoch in range(1, checkpoint.config.training.num_epochs + 1):
            total_step, loss_buffer = _train_epoch(
                epoch,
                total_step,
                loss_buffer,
                accelerator,
                encoder_to_embedder,
                backbone,
                task_layer,
                criterion,
                train_dataloader,
                train_transformed_artifacts,
                optimizer,
                scheduler,
                checkpoint,
                dataset,
                tracker,
            )
            if early_stopping is not None:
                _save_model(epoch, cache_dir, accelerator, encoder_to_embedder, backbone, task_layer)
            metrics = _test_epoch(
                epoch,
                cache_dir,
                accelerator,
                encoder_to_embedder,
                backbone,
                task_layer,
                head,
                evaluator,
                test_dataloader,
                test_transformed_artifacts,
                checkpoint,
                dataset,
                tracker,
            )
            if early_stopping is not None and early_stopping.should_stop(metrics):
                break

        _unwrap(accelerator, encoder_to_embedder, backbone, task_layer)
        if early_stopping is not None:
            _load_model(early_stopping.best_epoch, cache_dir, encoder_to_embedder, backbone, task_layer)
        accelerator.wait_for_everyone()

    return core.checkpoint.Trained(
        checkpoint.config,
        checkpoint.encoder_to_preprocessor,
        encoder_to_embedder,
        backbone,
        head,
    )


def _init_embedders(
    accelerator: Accelerator,
    checkpoint: core.checkpoint.Prepared,
) -> dict[str, core.encoders.Embedder]:
    encoder_to_embedder = {
        encoder: core.encoders.registry[desc.type].embedder.init(
            checkpoint.encoder_to_preprocessor[encoder],
            checkpoint.config.backbone.dim,
            **desc.embedder,
        )
        for encoder, desc in checkpoint.config.encoders.items()
    }
    for embedder in encoder_to_embedder.values():
        embedder.wrap(accelerator)
    return encoder_to_embedder


def _init_backbone(accelerator: Accelerator, checkpoint: core.checkpoint.Prepared) -> core.Backbone:
    history_aggregator_config = {
        "max_events_per_sequence": checkpoint.config.max_events_per_sequence,
        **checkpoint.config.backbone.history_aggregator.model_dump(),
    }
    event_aggregator_config = {
        "num_features": len(checkpoint.config.feature_to_encoder(event=True)),
        **checkpoint.config.backbone.event_aggregator.model_dump(),
    }
    if checkpoint.config.backbone.context_aggregator is not None:
        context_aggregator_config = {
            "num_features": len(checkpoint.config.feature_to_encoder(context=True)),
            **checkpoint.config.backbone.context_aggregator.model_dump(),
        }
    else:
        context_aggregator_config = None
    backbone = core.Backbone(
        checkpoint.config.backbone.dim,
        history_aggregator_config=history_aggregator_config,
        event_aggregator_config=event_aggregator_config,
        context_aggregator_config=context_aggregator_config,
    )
    backbone.wrap(accelerator)
    return backbone


def _init_task_entities(
    accelerator: Accelerator,
    checkpoint: core.checkpoint.Prepared,
    dataset: core.Dataset,
) -> tuple[core.tasks.Layer, core.tasks.Criterion, core.tasks.Head, core.tasks.Evaluator]:
    task_layer = core.tasks.registry[checkpoint.config.task.type].layer.init(
        dataset.task_preprocessor,
        checkpoint.config.backbone.dim,
        **checkpoint.config.task.layer,
    )
    task_layer.wrap(accelerator)

    criterion = core.tasks.registry[checkpoint.config.task.type].criterion.init(
        dataset.task_preprocessor,
        task_layer,
        **checkpoint.config.task.criterion,
    )
    criterion = accelerator.prepare_model(criterion)

    head = core.tasks.registry[checkpoint.config.task.type].head.init(
        dataset.task_preprocessor,
        task_layer,
        **checkpoint.config.task.head,
    )

    metric_type_to_cls = core.tasks.registry[checkpoint.config.task.type].metrics
    name_to_metric = {
        name: metric_type_to_cls[metric_desc.type](**metric_desc.params)
        for name, metric_desc in checkpoint.config.task.metrics.items()
    }
    evaluator = core.tasks.registry[checkpoint.config.task.type].evaluator(name_to_metric)

    return task_layer, criterion, head, evaluator


def _log_parameters(
    encoder_to_embedder: dict[str, core.encoders.Embedder],
    backbone: core.Backbone,
    task_layer: core.tasks.Layer,
    criterion: core.tasks.Criterion,
) -> None:
    encoder_num_params_total = 0
    for encoder, embedder in encoder_to_embedder.items():
        encoder_num_params = sum(p.numel() for p in embedder.parameters() if p.requires_grad)
        logger.info("encoder %s has %s trainable parameters", encoder, encoder_num_params)
        encoder_num_params_total += encoder_num_params
    logger.info("encoders has %s trainable parameters in total", encoder_num_params_total)

    backbone_num_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    logger.info("backbone has %s trainable parameters", backbone_num_params)

    task_layer_num_params = sum(p.numel() for p in task_layer.parameters() if p.requires_grad)
    logger.info("task layer has %s trainable parameters", task_layer_num_params)

    total_model_num_params = encoder_num_params_total + backbone_num_params + task_layer_num_params
    logger.info("model has %s trainable parameters in total", total_model_num_params)

    criterion_num_params = sum(p.numel() for p in criterion.parameters() if p.requires_grad)
    logger.info("criterion has %s trainable parameters", criterion_num_params)


def _create_optimizer(
    accelerator: Accelerator,
    encoder_to_embedder: dict[str, core.encoders.Embedder],
    backbone: core.Backbone,
    task_layer: core.tasks.Layer,
    criterion: core.tasks.Criterion,
    checkpoint: core.checkpoint.Prepared,
) -> optim.Optimizer:
    optimizer = core.optimization.optimizers.registry[checkpoint.config.training.optimizer.type](
        it.chain(
            it.chain(*[embedder.parameters() for embedder in encoder_to_embedder.values()]),
            backbone.parameters(),
            task_layer.parameters(),
            criterion.parameters(),
        ),
        **checkpoint.config.training.optimizer.params,
    )
    return accelerator.prepare_optimizer(optimizer)


def _create_scheduler(
    accelerator: Accelerator,
    optimizer: optim.Optimizer,
    checkpoint: core.checkpoint.Prepared,
) -> optim.lr_scheduler.LRScheduler:
    scheduler = core.optimization.schedulers.registry[checkpoint.config.training.scheduler.type](
        optimizer,
        **checkpoint.config.training.scheduler.params,
    )
    return accelerator.prepare_scheduler(scheduler)


def _create_early_stopping(checkpoint: core.checkpoint.Prepared) -> core.optimization.EarlyStopping | None:
    if checkpoint.config.training.early_stopping is None:
        return None
    return core.optimization.EarlyStopping(**checkpoint.config.training.early_stopping.model_dump())


def _create_dataloaders(
    accelerator: Accelerator,
    checkpoint: core.checkpoint.Prepared,
    dataset: core.Dataset,
) -> tuple[DataLoader, DataLoader]:
    event_feature_to_preprocessor = {
        feature: checkpoint.encoder_to_preprocessor[encoder]
        for feature, encoder in checkpoint.config.feature_to_encoder(event=True).items()
    }
    context_feature_to_preprocessor = {
        feature: checkpoint.encoder_to_preprocessor[encoder]
        for feature, encoder in checkpoint.config.feature_to_encoder(context=True).items()
    }
    events_reader = dataset.event_store.open_reader(checkpoint.config.max_events_per_sequence)

    train_dataloader = core.dataloaders.samples(
        dataset.train_samples,
        events_reader,
        event_feature_to_preprocessor,
        context_feature_to_preprocessor,
        task_preprocessor=dataset.task_preprocessor,
        batch_size=checkpoint.config.training.dataloader.batch_size,
        shuffle=checkpoint.config.training.dataloader.shuffle,
        num_workers=checkpoint.config.training.dataloader.num_workers,
        pin_memory=checkpoint.config.training.dataloader.pin_memory,
        persistent_workers=checkpoint.config.training.dataloader.num_workers > 0,
    )
    train_dataloader = accelerator.prepare_data_loader(train_dataloader)

    dataset.test_samples = dataset.test_samples.with_row_index("_index")
    test_dataloader = core.dataloaders.samples(
        dataset.test_samples,
        events_reader,
        event_feature_to_preprocessor,
        context_feature_to_preprocessor,
        extras=["_index", *core.tasks.registry[checkpoint.config.task.type].head.extras],
        batch_size=checkpoint.config.training.test_dataloader.batch_size,
        shuffle=False,
        num_workers=checkpoint.config.training.test_dataloader.num_workers,
        pin_memory=checkpoint.config.training.test_dataloader.pin_memory,
        persistent_workers=checkpoint.config.training.test_dataloader.num_workers > 0,
    )
    test_dataloader = accelerator.prepare_data_loader(test_dataloader)

    return train_dataloader, test_dataloader


def _dynamic_transform_artifacts(checkpoint: core.checkpoint.Prepared, dataset: core.Dataset) -> tuple[t.Any, t.Any]:
    artifacts_feature_to_preprocessor = {
        feature: checkpoint.encoder_to_preprocessor[encoder]
        for feature, encoder in checkpoint.config.feature_to_encoder(artifacts=True).items()
    }
    train_transformed_artifacts = dataset.train_artifacts.dynamic_transform(artifacts_feature_to_preprocessor)
    test_transformed_artifacts = dataset.test_artifacts.dynamic_transform(artifacts_feature_to_preprocessor)
    return train_transformed_artifacts, test_transformed_artifacts


def _train_epoch(
    epoch: int,
    total_step: int,
    loss_buffer: list[torch.Tensor],
    accelerator: Accelerator,
    encoder_to_embedder: dict[str, core.encoders.Embedder],
    backbone: core.Backbone,
    task_layer: core.tasks.Layer,
    criterion: core.tasks.Criterion,
    train_dataloader: DataLoader,
    train_transformed_artifacts: t.Any,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    checkpoint: core.checkpoint.Prepared,
    dataset: core.Dataset,
    tracker: core.trackers.Tracker | None,
) -> tuple[int, list[torch.Tensor]]:
    accelerator.wait_for_everyone()
    for embedder in encoder_to_embedder.values():
        embedder.train()
    backbone.train()
    task_layer.train()

    backbone_feature_to_embedder = {
        feature: encoder_to_embedder[encoder]
        for feature, encoder in checkpoint.config.feature_to_encoder(event=True, context=True).items()
    }
    artifacts_feature_to_embedder = {
        feature: encoder_to_embedder[encoder]
        for feature, encoder in checkpoint.config.feature_to_encoder(artifacts=True).items()
    }
    for batch in tqdm(train_dataloader, desc=f"train {epoch = } / {checkpoint.config.training.num_epochs}"):
        with accelerator.accumulate(*encoder_to_embedder.values(), backbone, task_layer, criterion):
            with accelerator.autocast():
                backbone_embeddings = backbone(
                    backbone_feature_to_embedder,
                    batch.events_features,
                    batch.events_positions,
                    batch.events_timestamps,
                    batch.sample_timestamp,
                    batch.context_features,
                )
                embedded_artifacts = dataset.train_artifacts.embed(
                    train_transformed_artifacts,
                    artifacts_feature_to_embedder,
                    batch.target,
                )
                loss = criterion(backbone_embeddings, embedded_artifacts, batch.target)

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_step += 1
        loss_buffer.append(loss.detach())
        if accelerator.is_main_process and len(loss_buffer) == checkpoint.config.training.log_every_n_train_steps:
            tracker.log_metrics(
                {
                    "loss": {"train": torch.stack(loss_buffer).mean().item()},
                    "lr": {"train": scheduler.get_last_lr()[0]},
                },
                iteration=total_step,
            )
            loss_buffer.clear()
    accelerator.wait_for_everyone()

    return total_step, loss_buffer


def _save_model(
    epoch: int,
    cache_dir: Path,
    accelerator: Accelerator,
    encoder_to_embedder: dict[str, core.encoders.Embedder],
    backbone: core.Backbone,
    task_layer: core.tasks.Layer,
) -> None:
    accelerator.wait_for_everyone()
    state_dir = cache_dir / "states" / f"{epoch=:04}"
    for encoder, embedder in encoder_to_embedder.items():
        if sum(p.numel() for p in embedder.state_dict().values()) > 0:
            accelerator.save_model(embedder, state_dir / "encoders" / encoder)
    if sum(p.numel() for p in backbone.state_dict().values()) > 0:
        accelerator.save_model(backbone, state_dir / "backbone")
    if sum(p.numel() for p in task_layer.state_dict().values()) > 0:
        accelerator.save_model(task_layer, state_dir / "task_layer")


@torch.no_grad()
def _test_epoch(
    epoch: int,
    cache_dir: Path,
    accelerator: Accelerator,
    encoder_to_embedder: dict[str, core.encoders.Embedder],
    backbone: core.Backbone,
    task_layer: core.tasks.Layer,
    head: core.tasks.Head,
    evaluator: core.tasks.Evaluator,
    test_dataloader: DataLoader,
    test_transformed_artifacts: t.Any,
    checkpoint: core.checkpoint.Prepared,
    dataset: core.Dataset,
    tracker: core.trackers.Tracker | None,
) -> dict[str, dict[str, float]]:
    accelerator.wait_for_everyone()
    for embedder in encoder_to_embedder.values():
        embedder.eval()
    backbone.eval()
    task_layer.eval()

    backbone_feature_to_embedder = {
        feature: encoder_to_embedder[encoder]
        for feature, encoder in checkpoint.config.feature_to_encoder(event=True, context=True).items()
    }
    artifacts_feature_to_embedder = {
        feature: encoder_to_embedder[encoder]
        for feature, encoder in checkpoint.config.feature_to_encoder(artifacts=True).items()
    }

    with accelerator.autocast():
        embedded_artifacts = dataset.test_artifacts.embed(
            test_transformed_artifacts,
            artifacts_feature_to_embedder,
        )

    predictions = pl.DataFrame()
    for batch in tqdm(test_dataloader, desc=f"test {epoch = } / {checkpoint.config.training.num_epochs}"):
        with accelerator.autocast():
            backbone_embeddings = backbone(
                backbone_feature_to_embedder,
                batch.events_features,
                batch.events_positions,
                batch.events_timestamps,
                batch.sample_timestamp,
                batch.context_features,
            )
            prediction = head.predict(
                backbone_embeddings,
                embedded_artifacts,
                batch.extras,
                **evaluator.predict_kwargs,
            )
        predictions.vstack(
            pl.DataFrame(
                {
                    "_index": pl.Series("_index", batch.extras["_index"], dtype=pl.UInt32),
                    "prediction": prediction,
                },
            ),
            in_place=True,
        )
    predictions_dir = cache_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(predictions_dir / f"rank={accelerator.local_process_index}.pq")

    accelerator.wait_for_everyone()
    metrics = evaluator.calculate_metrics(
        dataset.test_samples.lazy()
        .join(pl.scan_parquet(predictions_dir), how="inner", on="_index")
        .unique("_index")
        .drop("_index")
        .collect(),
        dataset.test_artifacts,
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        shutil.rmtree(predictions_dir)
        tracker.log_metrics(metrics, iteration=epoch)
    accelerator.wait_for_everyone()
    return metrics


def _unwrap(
    accelerator: Accelerator,
    encoder_to_embedder: dict[str, core.encoders.Embedder],
    backbone: core.Backbone,
    task_layer: core.tasks.Layer,
) -> None:
    for embedder in encoder_to_embedder.values():
        embedder.unwrap(accelerator)
    backbone.unwrap(accelerator)
    task_layer.unwrap(accelerator)


def _load_model(
    epoch: int,
    cache_dir: Path,
    encoder_to_embedder: dict[str, core.encoders.Embedder],
    backbone: core.Backbone,
    task_layer: core.tasks.Layer,
) -> None:
    state_dir = cache_dir / "states" / f"{epoch=:04}"
    for encoder, embedder in encoder_to_embedder.items():
        if sum(p.numel() for p in embedder.state_dict().values()) > 0:
            load_checkpoint_in_model(embedder, state_dir / "encoders" / encoder)
    if sum(p.numel() for p in backbone.state_dict().values()) > 0:
        load_checkpoint_in_model(backbone, state_dir / "backbone")
    if sum(p.numel() for p in task_layer.state_dict().values()) > 0:
        load_checkpoint_in_model(task_layer, state_dir / "task_layer")
