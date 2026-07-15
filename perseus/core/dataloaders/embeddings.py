import typing as t
from collections import defaultdict

import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset


def embeddings(
    samples: pl.DataFrame,
    extras: list[str] | None = None,
    **kwargs: t.Any,
) -> DataLoader:
    return DataLoader(
        _EmbeddingsDataset(samples=samples, extras=extras),
        collate_fn=_collate_embeddings,
        **kwargs,
    )


class _Sample(t.NamedTuple):
    embeddings: torch.Tensor
    extras: dict[str, t.Any] | None


class _EmbeddingsDataset(Dataset):
    def __init__(self, *, samples: pl.DataFrame, extras: list[str] | None = None) -> None:
        super().__init__()

        self.samples = samples
        self.extras = extras

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> _Sample:
        sample = self.samples[index]

        embeddings = sample["embeddings"].to_torch()[0]

        if self.extras:
            extras = {extra: sample[extra].item() for extra in self.extras or [] if extra in sample.schema}
        else:
            extras = None

        return _Sample(embeddings, extras)


class _CollatedSamples(t.NamedTuple):
    embeddings: torch.Tensor
    extras: dict[str, t.Any] | None


def _collate_embeddings(samples: list[_Sample], /) -> _CollatedSamples:
    embeddings_chunks = []
    extras_chunks = defaultdict(list)
    for sample in samples:
        embeddings_chunks.append(sample.embeddings)

        if sample.extras:
            for name, value in sample.extras.items():
                extras_chunks[name].append(value)

    embeddings = torch.stack(embeddings_chunks)

    extras = dict(extras_chunks) if extras_chunks else None

    return _CollatedSamples(embeddings, extras)
