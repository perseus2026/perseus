import logging
import typing as t

import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from perseus import core, utils

logger = logging.getLogger(__name__)


def make_head_predictions(
    samples: pl.DataFrame,
    embeddings: pl.DataFrame,
    checkpoint: core.checkpoint.Trained,
    /,
    *,
    embedded_artifacts: t.Any | None = None,
) -> pl.DataFrame:
    dataloader = _create_dataloader(samples, embeddings, checkpoint)
    predictions = _make_predictions(dataloader, embedded_artifacts, checkpoint)
    return _join_predictions(samples, predictions)


def _create_dataloader(
    samples: pl.DataFrame,
    embeddings: pl.DataFrame,
    checkpoint: core.checkpoint.Trained,
) -> DataLoader:
    return core.dataloaders.embeddings(
        samples.join(embeddings, how="inner", on="_index").select(
            "_index",
            "embeddings",
            *[extra for extra in checkpoint.head.extras if extra in samples.schema],
        ),
        extras=["_index", *checkpoint.head.extras],
        batch_size=checkpoint.config.inference.head_dataloader.batch_size,
        shuffle=False,
        num_workers=checkpoint.config.inference.head_dataloader.num_workers,
        pin_memory=checkpoint.config.inference.head_dataloader.pin_memory,
    )


@torch.no_grad()
def _make_predictions(
    dataloader: DataLoader,
    embedded_artifacts: t.Any,
    checkpoint: core.checkpoint.Trained,
) -> pl.DataFrame:
    for embedder in checkpoint.encoder_to_embedder.values():
        embedder.eval().to(utils.torch.device, non_blocking=True)

    predictions = pl.DataFrame()
    for batch in tqdm(dataloader, desc="inference head"):
        with torch.autocast(utils.torch.device.type):
            prediction = checkpoint.head.predict(
                batch.embeddings.to(utils.torch.device, non_blocking=True),
                embedded_artifacts,
                batch.extras,
                **checkpoint.config.inference.predict_kwargs,
            )
        predictions.vstack(
            pl.DataFrame(
                {
                    "_index": pl.Series("_index", batch.extras["_index"], dtype=pl.UInt32()),
                    "prediction": prediction,
                },
            ),
            in_place=True,
        )
    return predictions.rechunk()


def _join_predictions(samples: pl.DataFrame, predictions: pl.DataFrame) -> pl.DataFrame:
    return samples.join(predictions, how="inner", on="_index", maintain_order="left")
