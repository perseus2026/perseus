import typing as t
from pathlib import Path

import polars as pl
import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from torch import nn

from perseus.core.encoders import base


class Observer(base.Observer):
    def __init__(self) -> None:
        self.texts = pl.Series("text", dtype=pl.String())

    def observe(self, values: pl.Series, /) -> None:
        self.texts = pl.concat([self.texts, values]).unique().drop_nulls()


class Preprocessor(base.Preprocessor):
    def __init__(self, tokenizer: Tokenizer, /) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def fit(
        cls,
        observer: Observer,
        /,
        *,
        vocab_size: int = 30_000,
        min_frequency: int = 0,
        max_token_length: int | None = None,
        max_tokens_per_text: int = 512,
    ) -> t.Self:
        tokenizer = Tokenizer(models.BPE())
        tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.Strip(),
                normalizers.NFKD(),
                normalizers.StripAccents(),
                normalizers.Lowercase(),
            ],
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.train_from_iterator(
            observer.texts.to_list(),
            trainer=trainers.BpeTrainer(
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                max_token_length=max_token_length,
                special_tokens=["[PAD]"],  # used only to book idx=0 as padding_idx
            ),
        )
        tokenizer.enable_truncation(max_tokens_per_text)
        return cls(tokenizer)

    def static_transform(self, values: pl.Series, /) -> pl.Series:
        return (
            values.to_frame("text")
            .lazy()
            .join(
                (uniq_texts := values.unique().drop_nulls())
                .to_frame("text")
                .with_columns(
                    token_ids=pl.Series(
                        values=[tokens.ids for tokens in self.tokenizer.encode_batch_fast(uniq_texts.to_list())],
                        dtype=pl.List(pl.UInt32()),
                    ),
                )
                .lazy(),
                how="left",
                on="text",
                maintain_order="left",
            )
            .select(pl.col("token_ids").fill_null([]))
            .collect()
            .to_series()  # ty: ignore[unresolved-attribute]
        )

    def dynamic_transform(self, values: pl.Series, /) -> torch.Tensor:
        # TODO: use polars list padding in new version https://github.com/pola-rs/polars/pull/23323
        token_ids = (
            values.to_frame("token_ids")
            .lazy()
            .with_columns(num_tokens=pl.col("token_ids").list.len())
            .select(
                token_ids=pl.col("token_ids").list.concat(
                    pl.lit(0, dtype=pl.UInt32()).repeat_by(pl.col("num_tokens").max().clip(1) - pl.col("num_tokens")),
                ),
            )
            .collect()
            .to_series()
        )
        return token_ids.list.to_array(len(token_ids[0])).to_torch().to(torch.int32)

    def save(self, path: Path, /) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path / "tokenizer.json"))

    @classmethod
    def load(cls, path: Path, /) -> t.Self:
        return cls(Tokenizer.from_file(str(path / "tokenizer.json")))


class Embedder(base.Embedder):
    def __init__(self, num: int, dim: int, /) -> None:
        super().__init__()

        self.embeddings = nn.Embedding(num + 1, dim, padding_idx=0)

    @classmethod
    def init(cls, preprocessor: Preprocessor, dim: int, /) -> t.Self:
        return cls(preprocessor.tokenizer.get_vocab_size(), dim)

    def forward(self, indices: torch.Tensor, /) -> torch.Tensor:
        embeddings_sum = self.embeddings(indices).sum(dim=-2)
        is_non_padded_sum = (indices != 0).sum(dim=-1).to(torch.float32)
        return embeddings_sum / (is_non_padded_sum.unsqueeze(dim=-1) + 1e-6)


encoder = base.Encoder(Observer, Preprocessor, Embedder)
