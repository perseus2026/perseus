"""Builders for isolated smoke tests of the api stages.

Assemble a synthetic event_hub source (parquet, partitioned with the same murmur hash as the
pipeline) plus ready-made Prepared/Trained/Dataset for a classification task.
"""

import datetime as dt
from pathlib import Path

import polars as pl
from mmh3 import hash as murmur_hash

from perseus.core import checkpoint as ckpt
from perseus.core.dataset import Dataset
from perseus.core.encoders.id import Embedder as IdEmbedder
from perseus.core.event_hub import PARTITIONS_NUM, PARTITIONS_SEED, distribute_to_partitions
from perseus.core.event_store import EventStore
from perseus.core.tasks.classification.target import Artifacts as ClsArtifacts
from perseus.core.tasks.classification.target import Head as ClsHead
from perseus.core.tasks.classification.target import Layer as ClsLayer
from tests.unit.api.conftest import (
    DIM,
    make_backbone,
    make_config,
    make_id_preprocessor,
    make_task_preprocessor,
)

EVENT_DATE = dt.datetime(2024, 1, 1)
SAMPLE_DATE = dt.datetime(2024, 2, 1)


def partition_of(client_id: str) -> int:
    return murmur_hash(client_id, seed=PARTITIONS_SEED, signed=False) % PARTITIONS_NUM + 1


def setup_purchase_source(root: Path, clients: list[str]) -> None:
    """Distributes purchase events across the source's partitioned parquet files."""
    events = pl.DataFrame(
        {
            "client_id": [c for c in clients for _ in range(2)],
            "timestamp": [EVENT_DATE, EVENT_DATE + dt.timedelta(days=1)] * len(clients),
        },
    )
    date_dir = root / "purchase" / "2024-01-01"
    date_dir.mkdir(parents=True)
    with_pid = distribute_to_partitions(events)
    for (pid,), group in with_pid.group_by("partition_id"):
        group.drop("partition_id").write_parquet(date_dir / f"{pid:04}.pq")


def build_prepared() -> ckpt.Prepared:
    return ckpt.Prepared(make_config(), {"event": make_id_preprocessor()})


def build_trained() -> ckpt.Trained:
    cls_pre = make_task_preprocessor()
    head = ClsHead.init(cls_pre, ClsLayer.init(cls_pre, DIM))
    return ckpt.Trained(
        make_config(),
        {"event": make_id_preprocessor()},
        {"event": IdEmbedder(10, DIM)},
        make_backbone().eval(),
        head,
    )


def build_dataset(tmp_path: Path) -> Dataset:
    """Dataset with a disk event store: train c1,c2 / test c3, events are integer indices."""
    store = EventStore(tmp_path / "events", backend="disk")
    with store.open_writer() as writer:
        writer.write(
            pl.DataFrame(
                {
                    "client_id": ["c1", "c1", "c2", "c3"],
                    "timestamp": [EVENT_DATE, EVENT_DATE + dt.timedelta(days=1), EVENT_DATE, EVENT_DATE],
                    "position": [1, 2, 1, 1],
                    "event": [1, 2, 3, 4],
                },
            ),
        )
    return Dataset(
        train_samples=pl.DataFrame({"client_id": ["c1", "c2"], "timestamp": [SAMPLE_DATE] * 2, "target": [0, 1]}),
        train_artifacts=ClsArtifacts(),
        test_samples=pl.DataFrame({"client_id": ["c3"], "timestamp": [SAMPLE_DATE], "target": ["a"]}),
        test_artifacts=ClsArtifacts(),
        event_store=store,
        task_preprocessor=make_task_preprocessor(),
    )
