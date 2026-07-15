import logging
import typing as t

import polars as pl
import torch

from perseus import core, utils

logger = logging.getLogger(__name__)


def embed_artifacts(
    artifacts: tuple[t.Any, ...] | dict[str, t.Any] | core.tasks.Artifacts | None,
    checkpoint: core.checkpoint.Trained,
    /,
) -> pl.DataFrame:
    artifacts = _init(artifacts, checkpoint)
    return _embed(artifacts, checkpoint)


def _init(
    artifacts: tuple[t.Any, ...] | dict[str, t.Any] | core.tasks.Artifacts | None,
    checkpoint: core.checkpoint.Trained,
) -> core.tasks.Artifacts:
    artifacts_cls = core.tasks.registry[checkpoint.config.task.type].artifacts
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


@torch.no_grad()
def _embed(artifacts: core.tasks.Artifacts, checkpoint: core.checkpoint.Trained) -> t.Any:
    feature_to_preprocessor = {
        feature: checkpoint.encoder_to_preprocessor[encoder]
        for feature, encoder in checkpoint.config.feature_to_encoder(artifacts=True).items()
    }
    artifacts = artifacts.static_transform(feature_to_preprocessor)
    transformed_artifacts = artifacts.dynamic_transform(feature_to_preprocessor)

    for embedder in checkpoint.encoder_to_embedder.values():
        embedder.eval().to(utils.torch.device, non_blocking=True)
    feature_to_embedder = {
        feature: checkpoint.encoder_to_embedder[encoder]
        for feature, encoder in checkpoint.config.feature_to_encoder(artifacts=True).items()
    }
    return artifacts.embed(transformed_artifacts, feature_to_embedder)
