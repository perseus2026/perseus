import datetime as dt
import logging

import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from perseus import core, utils

logger = logging.getLogger(__name__)


def make_backbone_embeddings(
    partition_id: int,
    samples: pl.DataFrame,
    checkpoint: core.checkpoint.Trained,
    /,
) -> pl.DataFrame | None:
    samples = _prepare_samples(samples)

    event_hub = core.event_hub.Reader(checkpoint.config)
    events = event_hub.read(partition_id, samples.select("client_id", _sample_timestamp="timestamp"))
    if events is None:
        return None

    samples = _filter_samples(samples, events)

    events = _static_transform_events(events, checkpoint)
    samples = _static_transform_samples(samples, checkpoint)

    event_store = core.EventStore(backend=checkpoint.config.inference.event_store_backend)
    with event_store.open_writer() as writer:
        writer.write(events)

    dataloader = _create_dataloader(
        samples,
        event_store.open_reader(checkpoint.config.max_events_per_sequence),
        checkpoint,
    )
    return _make_embeddings(dataloader, checkpoint)


def _prepare_samples(samples: pl.DataFrame) -> pl.DataFrame:
    if "timestamp" not in samples.schema:
        samples = samples.with_columns(
            timestamp=pl.lit(
                dt.datetime.now(dt.timezone(dt.timedelta(hours=0), "UTC")).replace(tzinfo=None),
                dtype=pl.Datetime("ns"),
            ),
        )
    return samples.select("_index", "client_id", "timestamp", *(["context"] if "context" in samples.schema else []))


@utils.timer(logger, "filtering samples")
def _filter_samples(samples: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    return (
        samples.lazy()
        .join(
            events.lazy().group_by("client_id").agg(_first_event_timestamp=pl.col("timestamp").min()),
            how="inner",
            on="client_id",
        )
        .filter(pl.col("_first_event_timestamp") < pl.col("timestamp"))
        .drop("_first_event_timestamp")
        .collect()
    )


@utils.timer(logger, "static transforming events")
def _static_transform_events(events: pl.DataFrame, checkpoint: core.checkpoint.Trained) -> pl.DataFrame:
    return events.with_columns(
        pl.col(feature).map_batches(checkpoint.encoder_to_preprocessor[encoder].static_transform)
        for feature, encoder in checkpoint.config.feature_to_encoder(event=True).items()
    )


@utils.timer(logger, "static transforming samples")
def _static_transform_samples(samples: pl.DataFrame, checkpoint: core.checkpoint.Trained) -> pl.DataFrame:
    if context_feature_to_encoder := checkpoint.config.feature_to_encoder(context=True):
        samples = samples.with_columns(
            context=samples["context"]
            .struct.unnest()
            .with_columns(
                pl.col(feature).map_batches(checkpoint.encoder_to_preprocessor[encoder].static_transform)
                for feature, encoder in context_feature_to_encoder.items()
            )
            .to_struct(),
        )
    return samples


def _create_dataloader(
    samples: pl.DataFrame,
    events_reader: core.event_store.EventsReader,
    checkpoint: core.checkpoint.Trained,
) -> DataLoader:
    event_feature_to_preprocessor = {
        feature: checkpoint.encoder_to_preprocessor[encoder]
        for feature, encoder in checkpoint.config.feature_to_encoder(event=True).items()
    }
    context_feature_to_preprocessor = {
        feature: checkpoint.encoder_to_preprocessor[encoder]
        for feature, encoder in checkpoint.config.feature_to_encoder(context=True).items()
    }
    return core.dataloaders.samples(
        samples,
        events_reader,
        event_feature_to_preprocessor,
        context_feature_to_preprocessor,
        extras=["_index"],
        batch_size=checkpoint.config.inference.backbone_dataloader.batch_size,
        shuffle=False,
        num_workers=checkpoint.config.inference.backbone_dataloader.num_workers,
        pin_memory=checkpoint.config.inference.backbone_dataloader.pin_memory,
    )


@torch.no_grad()
def _make_embeddings(dataloader: DataLoader, checkpoint: core.checkpoint.Trained) -> pl.DataFrame:
    for embedder in checkpoint.encoder_to_embedder.values():
        embedder.eval().to(utils.torch.device, non_blocking=True)
    checkpoint.backbone.eval().to(utils.torch.device, non_blocking=True)

    feature_to_embedder = {
        feature: checkpoint.encoder_to_embedder[encoder]
        for feature, encoder in checkpoint.config.feature_to_encoder(event=True, context=True).items()
    }

    embeddings = pl.DataFrame()
    for batch in tqdm(dataloader, desc="inference backbone"):
        with torch.autocast(utils.torch.device.type):
            backbone_embeddings = checkpoint.backbone(
                feature_to_embedder,
                {
                    feature: values.to(utils.torch.device, non_blocking=True)
                    for feature, values in batch.events_features.items()
                },
                batch.events_positions.to(utils.torch.device, non_blocking=True),
                batch.events_timestamps.to(utils.torch.device, non_blocking=True),
                batch.sample_timestamp.to(utils.torch.device, non_blocking=True),
                {
                    feature: values.to(utils.torch.device, non_blocking=True)
                    for feature, values in batch.context_features.items()
                }
                if batch.context_features is not None
                else batch.context_features,
            )
        embeddings.vstack(
            pl.DataFrame(
                {
                    "_index": pl.Series("_index", batch.extras["_index"], dtype=pl.UInt32()),
                    "embeddings": pl.Series(
                        "embeddings",
                        backbone_embeddings.cpu().to(torch.float32).numpy(),
                        dtype=pl.Array(pl.Float32(), checkpoint.config.backbone.dim),
                    ),
                },
            ),
            in_place=True,
        )
    return embeddings.rechunk()
