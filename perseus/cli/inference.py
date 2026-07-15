from pathlib import Path

import polars as pl
import torch
from cyclopts import App
from structlog.contextvars import bound_contextvars

from perseus import api, core, utils

app = App()


@app.command
def distribute_samples(*, workdir: Path = Path("workdir/")) -> None:
    samples = pl.read_parquet(workdir / "basis" / "samples.pq")
    samples_dir = workdir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    for partition_id, partition_samples in api.distribute_samples(samples):
        partition_samples.write_parquet(samples_dir / core.event_hub.filename(partition_id))


@app.command
def make_backbone_embeddings(
    *,
    partitions_bin_idx: int = 1,
    partitions_num_bins: int = 1,
    overrides: utils.cyclopts.JsonMapping | None = None,
    workdir: Path = Path("workdir/"),
) -> None:
    checkpoint = core.checkpoint.Trained.load(workdir / "checkpoint", overrides=overrides)

    samples_dir = workdir / "samples"
    embeddings_dir = workdir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(samples_dir.iterdir()):
        partition_id = core.event_hub.partition_id(path.name)
        if partition_id % partitions_num_bins != partitions_bin_idx % partitions_num_bins:
            continue

        samples = pl.read_parquet(path)

        with bound_contextvars(partition_id=partition_id):
            embeddings = api.make_backbone_embeddings(partition_id, samples, checkpoint)
            if embeddings is None:
                continue

        embeddings.write_parquet(embeddings_dir / path.name)


@app.command
def make_items_embeddings(
    *,
    overrides: utils.cyclopts.JsonMapping | None = None,
    workdir: Path = Path("workdir/"),
) -> None:
    checkpoint = core.checkpoint.Trained.load(workdir / "checkpoint", overrides=overrides)

    artifacts = core.tasks.registry[checkpoint.config.task.type].artifacts.load(workdir / "basis" / "artifacts")
    items, item_features = api.embed_artifacts(artifacts, checkpoint)

    layer = checkpoint.head.layer
    layer.eval().to(utils.torch.device, non_blocking=True)
    with torch.autocast(utils.torch.device.type):
        embeddings = layer.item_aggregator(item_features.to(utils.torch.device, non_blocking=True))

    items_dir = workdir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "item": items,
            "embeddings": pl.Series(
                embeddings.cpu().to(torch.float32).numpy(),
                dtype=pl.Array(pl.Float32(), checkpoint.config.backbone.dim),
            ),
        },
    ).write_parquet(items_dir / "embeddings.pq")


@app.command
def make_head_predictions(
    *,
    partitions_bin_idx: int = 1,
    partitions_num_bins: int = 1,
    overrides: utils.cyclopts.JsonMapping | None = None,
    workdir: Path = Path("workdir/"),
) -> None:
    checkpoint = core.checkpoint.Trained.load(workdir / "checkpoint", overrides=overrides)

    artifacts = core.tasks.registry[checkpoint.config.task.type].artifacts.load(workdir / "basis" / "artifacts")
    embedded_artifacts = api.embed_artifacts(artifacts, checkpoint)

    samples_dir = workdir / "samples"
    embeddings_dir = workdir / "embeddings"
    predictions_dir = workdir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(samples_dir.iterdir()):
        partition_id = core.event_hub.partition_id(path.name)
        if partition_id % partitions_num_bins != partitions_bin_idx % partitions_num_bins:
            continue

        if not (embeddings_path := embeddings_dir / path.name).exists():
            continue

        samples = pl.read_parquet(path)
        embeddings = pl.read_parquet(embeddings_path)

        with bound_contextvars(partition_id=partition_id):
            predictions = api.make_head_predictions(
                samples,
                embeddings,
                checkpoint,
                embedded_artifacts=embedded_artifacts,
            )

        predictions.write_parquet(predictions_dir / path.name)
