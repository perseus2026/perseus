import logging
import typing as t
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
from structlog.contextvars import bound_contextvars

from perseus import core, utils

logger = logging.getLogger(__name__)


def prepare_dataset(
    config: core.Config,
    train_samples: pl.DataFrame,
    test_samples: pl.DataFrame,
    /,
    *,
    train_artifacts: tuple[t.Any, ...] | dict[str, t.Any] | core.tasks.Artifacts | None = None,
    test_artifacts: tuple[t.Any, ...] | dict[str, t.Any] | core.tasks.Artifacts | None = None,
) -> tuple[core.checkpoint.Prepared, core.Dataset]:
    train_artifacts = _init_artifacts(train_artifacts, config)
    test_artifacts = _init_artifacts(test_artifacts, config)

    encoder_to_observer = {
        encoder: core.encoders.registry[desc.type].observer() for encoder, desc in config.encoders.items()
    }
    event_hub = core.event_hub.Reader(config)
    event_store = core.EventStore(backend="disk")
    client_event_first_timestamp = pl.DataFrame()
    with TemporaryDirectory() as _cache_dir:
        cache_dir = Path(_cache_dir)
        for partition_id, samples in _distribute_samples(train_samples, test_samples):
            with bound_contextvars(partition_id=partition_id):
                if (events := event_hub.read(partition_id, samples)) is None:
                    continue
                _observe_events(events, samples, encoder_to_observer, config)
                events.write_parquet(cache_dir / core.event_hub.filename(partition_id))

                client_event_first_timestamp.vstack(
                    events.group_by("client_id").agg(_first_event_timestamp=pl.col("timestamp").min()),
                    in_place=True,
                )

        train_samples, test_samples = _filter_samples(client_event_first_timestamp, train_samples, test_samples)
        _observe_samples(train_samples, encoder_to_observer, config)
        task_preprocessor = _fit_task_preprocessor(train_samples, train_artifacts, encoder_to_observer, config)
        encoder_to_preprocessor = _fit_encoders_preprocessors(encoder_to_observer, config)

        with event_store.open_writer() as writer:
            for partition_path in sorted(cache_dir.iterdir()):
                partition_id = core.event_hub.partition_id(partition_path.name)
                with bound_contextvars(partition_id=partition_id):
                    events = pl.read_parquet(partition_path)
                    events = _static_transform_events(events, encoder_to_preprocessor, config)
                    writer.write(events)

    train_samples, train_artifacts, test_samples, test_artifacts = _static_transform_basis(
        train_samples,
        train_artifacts,
        test_samples,
        test_artifacts,
        encoder_to_preprocessor,
        task_preprocessor,
        config,
    )

    return (
        core.checkpoint.Prepared(config, encoder_to_preprocessor),
        core.Dataset(train_samples, train_artifacts, test_samples, test_artifacts, event_store, task_preprocessor),
    )


def _init_artifacts(
    artifacts: tuple[t.Any, ...] | dict[str, t.Any] | core.tasks.Artifacts | None,
    config: core.Config,
) -> core.tasks.Artifacts:
    artifacts_cls = core.tasks.registry[config.task.type].artifacts
    match artifacts:
        case tuple():
            return artifacts_cls(*artifacts)
        case dict():
            return artifacts_cls(**artifacts)
        case None:
            return artifacts_cls()
        case core.tasks.Artifacts() if not isinstance(artifacts, artifacts_cls):
            raise TypeError("artifacts has mismatching type with task")
        case core.tasks.Artifacts():
            return artifacts


def _distribute_samples(
    train_samples: pl.DataFrame,
    test_samples: pl.DataFrame,
) -> t.Generator[tuple[int, pl.DataFrame], None, None]:
    samples = pl.concat(
        [
            train_samples.select("client_id", _sample_timestamp="timestamp", _fold=pl.lit("train")),
            test_samples.select("client_id", _sample_timestamp="timestamp", _fold=pl.lit("test")),
        ],
    )
    samples = core.event_hub.distribute_to_partitions(samples).sort("partition_id")
    logger.info("has %s samples for %s partitions", len(samples), samples["partition_id"].n_unique())
    for (partition_id,), partition_samples in samples.group_by("partition_id", maintain_order=True):
        yield partition_id, partition_samples.drop("partition_id")


@utils.timer(logger, "observing events")
def _observe_events(
    events: pl.DataFrame,
    samples: pl.DataFrame,
    encoder_to_observer: dict[str, core.encoders.Observer],
    config: core.Config,
) -> None:
    train_events = (
        events.lazy()
        .join(
            samples.lazy()
            .filter(pl.col("_fold") == "train")
            .group_by("client_id")
            .agg(_last_sample_timestamp=pl.col("_sample_timestamp").max()),
            how="inner",
            on="client_id",
        )
        .filter(pl.col("timestamp") < pl.col("_last_sample_timestamp"))
        .drop("_last_sample_timestamp")
        .collect()
    )
    for feature, encoder in config.feature_to_encoder(event=True).items():
        encoder_to_observer[encoder].observe(train_events[feature])


@utils.timer(logger, "filtering samples")
def _filter_samples(
    client_event_first_timestamp: pl.DataFrame,
    train_samples: pl.DataFrame,
    test_samples: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    train_samples = (
        train_samples.lazy()
        .join(client_event_first_timestamp.lazy(), how="inner", on="client_id")
        .filter(pl.col("_first_event_timestamp") < pl.col("timestamp"))
        .drop("_first_event_timestamp")
        .collect()
    )
    test_samples = (
        test_samples.lazy()
        .join(client_event_first_timestamp.lazy(), how="inner", on="client_id")
        .filter(pl.col("_first_event_timestamp") < pl.col("timestamp"))
        .drop("_first_event_timestamp")
        .collect()
    )
    return train_samples, test_samples


@utils.timer(logger, "observing samples")
def _observe_samples(
    train_samples: pl.DataFrame,
    encoder_to_observer: dict[str, core.encoders.Observer],
    config: core.Config,
) -> None:
    if context_feature_to_encoder := config.feature_to_encoder(context=True):
        train_context = (
            train_samples.lazy()
            .select("client_id", "context")
            .unique()
            .select(pl.col("context").struct.unnest())
            .collect()
        )
        for feature, encoder in context_feature_to_encoder.items():
            encoder_to_observer[encoder].observe(train_context[feature])


@utils.timer(logger, "fitting task preprocessor")
def _fit_task_preprocessor(
    train_samples: pl.DataFrame,
    train_artifacts: pl.DataFrame,
    encoder_to_observer: dict[str, core.encoders.Observer],
    config: core.Config,
) -> core.tasks.Preprocessor:
    artifacts_feature_to_observer = {
        feature: encoder_to_observer[encoder] for feature, encoder in config.feature_to_encoder(artifacts=True).items()
    }
    return core.tasks.registry[config.task.type].preprocessor.fit(
        train_samples["target"],
        train_artifacts,
        artifacts_feature_to_observer,
    )


@utils.timer(logger, "fitting encoders preprocessors")
def _fit_encoders_preprocessors(
    encoder_to_observer: dict[str, core.encoders.Observer],
    config: core.Config,
) -> dict[str, core.encoders.Preprocessor]:
    return {
        encoder: core.encoders.registry[desc.type].preprocessor.fit(encoder_to_observer[encoder], **desc.preprocessor)
        for encoder, desc in config.encoders.items()
    }


@utils.timer(logger, "static transforming events")
def _static_transform_events(
    events: pl.DataFrame,
    encoder_to_preprocessor: dict[str, core.encoders.Preprocessor],
    config: core.Config,
) -> pl.DataFrame:
    return events.with_columns(
        pl.col(feature).map_batches(encoder_to_preprocessor[encoder].static_transform)
        for feature, encoder in config.feature_to_encoder(event=True).items()
    )


@utils.timer(logger, "static transforming basis")
def _static_transform_basis(
    train_samples: pl.DataFrame,
    train_artifacts: core.tasks.Artifacts,
    test_samples: pl.DataFrame,
    test_artifacts: core.tasks.Artifacts,
    encoder_to_preprocessor: dict[str, core.encoders.Preprocessor],
    task_preprocessor: core.tasks.Preprocessor,
    config: core.Config,
) -> tuple[pl.DataFrame, core.tasks.Artifacts, pl.DataFrame, core.tasks.Artifacts]:
    if context_feature_to_encoder := config.feature_to_encoder(context=True):
        train_samples = train_samples.with_columns(
            context=train_samples["context"]
            .struct.unnest()
            .with_columns(
                pl.col(feature).map_batches(encoder_to_preprocessor[encoder].static_transform)
                for feature, encoder in context_feature_to_encoder.items()
            )
            .to_struct(),
        )
        test_samples = test_samples.with_columns(
            context=test_samples["context"]
            .struct.unnest()
            .with_columns(
                pl.col(feature).map_batches(encoder_to_preprocessor[encoder].static_transform)
                for feature, encoder in context_feature_to_encoder.items()
            )
            .to_struct(),
        )

    artifacts_feature_to_preprocessor = {
        feature: encoder_to_preprocessor[encoder]
        for feature, encoder in config.feature_to_encoder(artifacts=True).items()
    }
    train_artifacts = train_artifacts.static_transform(artifacts_feature_to_preprocessor)
    test_artifacts = test_artifacts.static_transform(artifacts_feature_to_preprocessor)

    train_samples = train_samples.with_columns(pl.col("target").map_batches(task_preprocessor.static_transform))

    train_samples = train_samples.sort("timestamp", "client_id")
    test_samples = test_samples.sort("timestamp", "client_id")

    return train_samples, train_artifacts, test_samples, test_artifacts
