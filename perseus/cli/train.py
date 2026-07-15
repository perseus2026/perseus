from pathlib import Path

import polars as pl
from cyclopts import App

from perseus import api, core, utils

app = App()


@app.command
def prepare_dataset(
    *,
    overrides: utils.cyclopts.JsonMapping | None = None,
    workdir: Path = Path("workdir/"),
) -> None:
    config = core.Config.load(workdir / "config.yaml", overrides=overrides)

    artifacts_cls = core.tasks.registry[config.task.type].artifacts

    train_path = workdir / "basis" / "train"
    train_samples = pl.read_parquet(train_path / "samples.pq")
    train_artifacts = artifacts_cls.load(train_path / "artifacts")

    test_path = workdir / "basis" / "test"
    test_samples = pl.read_parquet(test_path / "samples.pq")
    test_artifacts = artifacts_cls.load(test_path / "artifacts")

    checkpoint, dataset = api.prepare_dataset(
        config,
        train_samples,
        test_samples,
        train_artifacts=train_artifacts,
        test_artifacts=test_artifacts,
    )

    checkpoint.save(workdir / "checkpoint")
    dataset.save(workdir / "dataset")


@app.command
def fit_model(
    *,
    overrides: utils.cyclopts.JsonMapping | None = None,
    workdir: Path = Path("workdir/"),
) -> None:
    checkpoint = core.checkpoint.Prepared.load(workdir / "checkpoint", overrides=overrides)
    dataset = core.Dataset.load(workdir / "dataset", checkpoint.config)

    checkpoint = api.fit_model(checkpoint, dataset)

    checkpoint.save(workdir / "checkpoint")
