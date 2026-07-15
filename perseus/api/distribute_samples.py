import logging
import typing as t

import polars as pl

from perseus import core

logger = logging.getLogger(__name__)


def distribute_samples(samples: pl.DataFrame, /) -> t.Generator[tuple[int, pl.DataFrame], None, None]:
    samples = core.event_hub.distribute_to_partitions(samples)
    logger.info("has %s samples for %s partitions", len(samples), samples["partition_id"].n_unique())
    for (partition_id,), partition_samples in samples.group_by("partition_id", maintain_order=True):
        yield partition_id, partition_samples.drop("partition_id").with_row_index("_index").rechunk()
