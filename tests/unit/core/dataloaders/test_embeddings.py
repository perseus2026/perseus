"""Tests for dataloaders.embeddings: dataset, collate, DataLoader."""

import polars as pl
import torch

from perseus.core.dataloaders.embeddings import _collate_embeddings, _EmbeddingsDataset, embeddings


def _samples() -> pl.DataFrame:
    # the embeddings column is an Array (fixed size), as in production (.to_torch requires Array)
    return pl.DataFrame(
        {"embeddings": [[1.0, 2.0], [3.0, 4.0]], "_index": [0, 1]},
        schema_overrides={"embeddings": pl.Array(pl.Float32, 2)},
    )


def test_dataset_getitem_returns_embedding_vector() -> None:
    dataset = _EmbeddingsDataset(samples=_samples(), extras=["_index"])
    sample = dataset[0]
    assert sample.embeddings.tolist() == [1.0, 2.0]
    assert sample.extras == {"_index": 0}
    assert len(dataset) == 2


def test_dataset_getitem_without_extras() -> None:
    sample = _EmbeddingsDataset(samples=_samples())[1]
    assert sample.embeddings.tolist() == [3.0, 4.0]
    assert sample.extras is None


def test_collate_stacks_embeddings_and_groups_extras() -> None:
    dataset = _EmbeddingsDataset(samples=_samples(), extras=["_index"])
    out = _collate_embeddings([dataset[0], dataset[1]])
    assert out.embeddings.shape == (2, 2)
    assert torch.allclose(out.embeddings, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert out.extras == {"_index": [0, 1]}


def test_dataloader_end_to_end() -> None:
    loader = embeddings(_samples(), extras=["_index"], batch_size=2)
    batch = next(iter(loader))
    assert batch.embeddings.shape == (2, 2)
    assert batch.extras == {"_index": [0, 1]}
