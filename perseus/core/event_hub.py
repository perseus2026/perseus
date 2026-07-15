import logging
import os
from functools import cached_property
from pathlib import Path

import polars as pl
from mmh3 import hash as murmur_hash
from tenacity import Retrying, stop_after_attempt

from perseus import utils
from perseus.core.config import Config

PARTITIONS_NUM = 100
PARTITIONS_SEED = 0
SOURCE_ENV_PREFIX = "internal_storage_"

logger = logging.getLogger(__name__)


def distribute_to_partitions[DFType: pl.DataFrame | pl.LazyFrame](samples: DFType, /) -> DFType:
    return samples.join(
        samples.select("client_id")
        .unique()
        .with_columns(
            partition_id=pl.col("client_id").map_elements(
                lambda x: murmur_hash(x, seed=PARTITIONS_SEED, signed=False) % PARTITIONS_NUM + 1,
                return_dtype=pl.UInt32(),
            ),
        ),
        how="inner",
        on="client_id",
    )


def filename(partition_id: int, /) -> str:
    return f"{partition_id:04}.pq"


def partition_id(filename: str, /) -> int:
    return int(filename[:-3])


def find_source_to_path() -> dict[str, Path]:
    prefix = SOURCE_ENV_PREFIX.lower()
    source_to_path: dict[str, Path] = {}
    for key, value in os.environ.items():
        key = key.lower()
        if not key.startswith(prefix):
            continue
        key = key.removeprefix(prefix)
        source_to_path[key] = Path(value)
    return source_to_path


class Reader:
    MAX_ATTEMPTS = 3

    def __init__(self, config: Config, /) -> None:
        self.source_to_path = find_source_to_path()
        existing_sources = set(self.source_to_path.keys())
        declared_sources = {event.source for event in config.events.values()}
        if needed_sources := declared_sources - existing_sources:
            raise RuntimeError(f"need to connect sources: {', '.join(needed_sources)}")

        self.config = config

    @utils.timer(logger, "reading events")
    def read(self, partition_id: int, samples: pl.DataFrame, /) -> pl.DataFrame | None:
        for attempt in Retrying(stop=stop_after_attempt(self.MAX_ATTEMPTS), reraise=True):
            with attempt:
                if (samples_with_events := self._join_events(partition_id, samples.lazy())) is None:
                    return None
        samples_with_events = self._filter_by_time(samples_with_events)
        samples_with_events = self._sort_by_priority(samples_with_events)
        samples_with_events = self._filter_by_count(samples_with_events)
        samples_with_events = samples_with_events.collect(engine="streaming")
        if samples_with_events.is_empty():
            logger.warning("not found samples with events")
            return None
        return self._flatten_unique_events_by_client(samples_with_events)

    def _join_events(self, partition_id: int, samples: pl.LazyFrame) -> pl.LazyFrame | None:
        events = []
        partition_file = filename(partition_id)
        for name, event in self.config.events.items():
            files = [
                date_partition_file
                for date_dir in sorted((self.source_to_path[event.source] / name).iterdir())
                if (date_partition_file := date_dir / partition_file).exists()
            ]
            if not files:
                logger.warning("not found %s events for partition_id = %s", name, partition_id)
                continue

            expr = pl.scan_parquet(files, extra_columns="ignore").select(
                "client_id",
                "timestamp",
                **self._event_to_attr_exprs[name],
                _priority=pl.lit(event.priority, pl.Int32()),
            )

            if multi_attrs := self._event_to_multi_attributes.get(name):
                expr = (
                    expr.with_columns(
                        _multi_attrs_indices=pl.int_ranges(
                            pl.max_horizontal(pl.col(attr).list.len().fill_null(0) for attr in multi_attrs),
                        )
                        .list.sample(fraction=1, shuffle=True)
                        .list.head(event.max_tokens),
                    )
                    .with_columns(
                        pl.when(pl.col(attr).is_not_null())
                        .then(pl.col(attr).list.gather("_multi_attrs_indices"))
                        .alias(attr)
                        for attr in multi_attrs
                    )
                    .drop("_multi_attrs_indices")
                )

            events.append(expr)

        if not events:
            logger.warning("no events found for partition_id = %s", partition_id)
            return None
        return samples.with_row_index("_sample_index").join(pl.concat(events), how="inner", on="client_id")

    @cached_property
    def _event_to_attr_exprs(self) -> dict[str, dict[str, pl.Expr]]:
        event_to_schema = {
            name: pl.scan_parquet(self.source_to_path[event.source] / name).collect_schema()
            for name, event in self.config.events.items()
        }

        attr_to_dtype = {}
        for event_name, event in self.config.events.items():
            for attr_name, attr in event.attributes.items():
                if attr_name == "event":
                    continue
                dtype = event_to_schema[event_name][attr.field]
                if attr_name in self._event_to_multi_attributes.get(event_name, []):
                    dtype = dtype.inner
                attr_to_dtype.setdefault(attr_name, dtype)
                if dtype != attr_to_dtype[attr_name]:
                    # TODO: add coercion (or just use concat relaxed?)
                    raise RuntimeError(f"found mismatched dtypes for attribute {attr_name}")

        event_to_attr_exprs = {}
        for event_name, event in self.config.events.items():
            attr_exprs = {"event": pl.lit(event_name)}
            for attr_name, dtype in attr_to_dtype.items():
                if self._attribute_to_multi[attr_name]:
                    dtype = pl.List(dtype)
                if (attr := event.attributes.get(attr_name)) is None:
                    attr_exprs[attr_name] = pl.lit(None, dtype=dtype)
                else:
                    attr_exprs[attr_name] = pl.col(attr.field).cast(dtype)
            event_to_attr_exprs[event_name] = attr_exprs
        return event_to_attr_exprs

    @cached_property
    def _event_to_multi_attributes(self) -> dict[str, list[str]]:
        return {
            name: multi_attributes
            for name, event in self.config.events.items()
            if (multi_attributes := [name for name, attr in event.attributes.items() if attr.multi])
        }

    @cached_property
    def _attribute_to_multi(self) -> dict[str, bool]:
        attribute_to_multi = {"event": False}
        for event in self.config.events.values():
            for name, attr in event.attributes.items():
                attribute_to_multi.setdefault(name, False)
                attribute_to_multi[name] = attribute_to_multi[name] or attr.multi
        return attribute_to_multi

    def _filter_by_time(self, samples_with_events: pl.LazyFrame) -> pl.LazyFrame:
        time_condition = (
            pl.col("timestamp").is_between(
                pl.col("_sample_timestamp").dt.offset_by(f"-{self.config.max_duration_per_sequence}"),
                pl.col("_sample_timestamp"),
                closed="left",
            )
            if self.config.max_duration_per_sequence
            else pl.col("timestamp") < pl.col("_sample_timestamp")
        )

        if self._event_to_max_duration:
            time_condition &= pl.col("event").is_in(self._event_to_max_duration.keys()).not_()
            for event, event_duration in self._event_to_max_duration.items():
                time_condition |= (pl.col("event") == event) & pl.col("timestamp").is_between(
                    pl.col("_sample_timestamp").dt.offset_by(f"-{event_duration}"),
                    pl.col("_sample_timestamp"),
                    closed="left",
                )

        return samples_with_events.filter(time_condition)

    @cached_property
    def _event_to_max_duration(self) -> dict[str, str]:
        return {
            name: event.max_duration_per_sequence
            for name, event in self.config.events.items()
            if event.max_duration_per_sequence
        }

    def _sort_by_priority(self, samples_with_events: pl.LazyFrame) -> pl.LazyFrame:
        return samples_with_events.sort(
            "_sample_index",
            *(["_priority"] if len({event.priority for event in self.config.events.values()}) > 1 else []),
            "timestamp",
        ).drop("_priority")

    def _filter_by_count(self, samples_with_events: pl.LazyFrame) -> pl.LazyFrame:
        if self._event_to_max_count:
            count_condition = pl.col("event").is_in(self._event_to_max_count.keys()).not_()
            for event, event_max_count in self._event_to_max_count.items():
                count_condition |= (pl.col("event") == event) & (
                    pl.int_range(pl.len(), 0, -1).over("_sample_index", "event") <= event_max_count
                )
            samples_with_events = samples_with_events.filter(count_condition)

        return samples_with_events.filter(
            pl.int_range(pl.len(), 0, -1).over("_sample_index") <= self.config.max_events_per_sequence,
        ).drop("_sample_index")

    @cached_property
    def _event_to_max_count(self) -> dict[str, int]:
        return {
            name: event.max_events_per_sequence
            for name, event in self.config.events.items()
            if event.max_events_per_sequence
        }

    def _flatten_unique_events_by_client(self, samples_with_events: pl.DataFrame) -> pl.DataFrame:
        all_columns = ["client_id", "timestamp", *list(self._attribute_to_multi.keys())]
        unique_columns = [col for col in all_columns if not self._attribute_to_multi.get(col)]
        events = (
            samples_with_events.lazy()
            .select(all_columns)
            .unique(unique_columns)
            .sort("client_id", "timestamp")
            .with_columns(position=pl.int_range(pl.len(), 0, -1).over("client_id"))
            .collect()
        )

        if self._event_to_multi_attributes:
            events = events.lazy()
            multi_attributes = {attr for attrs in self._event_to_multi_attributes.values() for attr in attrs}
            events_exploded = [
                events.filter(pl.col("event").is_in(self._event_to_multi_attributes.keys()).not_())
                .with_columns(pl.col(*multi_attributes).fill_null([None]))
                .explode(*multi_attributes),
            ]
            for event, event_multi_attributes in self._event_to_multi_attributes.items():
                event_exploded = events.filter(pl.col("event") == event).explode(*event_multi_attributes)
                if len(multi_attributes) > len(event_multi_attributes):
                    event_exploded = event_exploded.explode(*(multi_attributes - set(event_multi_attributes)))
                events_exploded.append(event_exploded)
            events = pl.concat(events_exploded).collect()

        return events.sort("client_id", "position", descending=[False, True])
