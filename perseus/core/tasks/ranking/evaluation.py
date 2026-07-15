import typing as t

import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from perseus.core.tasks import base
from perseus.core.tasks.ranking.target import Artifacts


class Evaluator(base.Evaluator):
    @property
    def predict_kwargs(self) -> dict[str, t.Any]:
        return {}

    def _prepare_samples(self, samples: pl.DataFrame, _artifacts: Artifacts, /) -> pl.DataFrame:
        samples = samples.with_row_index("qid").drop("items")
        return (
            samples.lazy()
            .drop("prediction")
            .explode("target")
            .unnest("target")
            .unnest("labels")
            .join(
                samples.lazy().select("qid", "prediction").explode("prediction").unnest("prediction").unnest("probas"),
                on=["qid", "item"],
                how="left",
                suffix="_prediction",
            )
            .with_columns(
                pl.col("score").fill_null(float("-inf")),
                pl.col(
                    *[
                        f"{field.name}_prediction"
                        for field in samples.schema["prediction"].inner.to_schema()["probas"].fields
                    ],
                ).fill_null(0.0),
            )
            .collect()
        )


class _SklearnScorer(t.Protocol):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs: t.Any) -> float: ...


class _ThresholdedSklearnMetric(base.Metric):
    _score_fn: t.ClassVar[_SklearnScorer]

    def __init__(self, *, label: str, threshold: float = 0.5) -> None:
        self.label = label
        self.threshold = threshold

    def calculate(self, samples: pl.DataFrame, _artifacts: Artifacts, /) -> float:
        score_fn = type(self)._score_fn  # noqa: SLF001
        y_true = samples[self.label].to_numpy()
        y_pred = samples[f"{self.label}_prediction"].to_numpy() > self.threshold
        return score_fn(y_true, y_pred)


class Accuracy(_ThresholdedSklearnMetric):
    _score_fn = accuracy_score


class Precision(_ThresholdedSklearnMetric):
    _score_fn = precision_score


class Recall(_ThresholdedSklearnMetric):
    _score_fn = recall_score


class F1(_ThresholdedSklearnMetric):
    _score_fn = f1_score


class _UnthresholdedSklearnMetric(base.Metric):
    _score_fn: t.ClassVar[_SklearnScorer]

    def __init__(self, *, label: str) -> None:
        self.label = label

    def calculate(self, samples: pl.DataFrame, _artifacts: Artifacts, /) -> float:
        score_fn = type(self)._score_fn  # noqa: SLF001
        y_true = samples[self.label].to_numpy()
        y_pred = samples[f"{self.label}_prediction"].to_numpy()
        return score_fn(y_true, y_pred)


class RocAuc(_UnthresholdedSklearnMetric):
    _score_fn = roc_auc_score


class PrAuc(_UnthresholdedSklearnMetric):
    _score_fn = average_precision_score


class AtKMetric(base.Metric):
    def __init__(self, *, k: int) -> None:
        super().__init__()

        self.k = k


class HitRateAtK(AtKMetric):
    def calculate(self, samples: pl.DataFrame, _artifacts: Artifacts, /) -> float:
        return (
            samples.lazy()
            .group_by("qid")
            .agg(
                value=(pl.col("relevance").sort_by("score", descending=True).head(self.k).max() > 0).cast(pl.Float32()),
            )
            .select("value")
            .mean()
            .collect()
            .item()
        )


class MrrAtK(AtKMetric):
    def calculate(self, samples: pl.DataFrame, _artifacts: Artifacts, /) -> float:
        relevant_in_top_k = pl.col("relevance").sort_by("score", descending=True).head(self.k) > 0
        return (
            samples.lazy()
            .group_by("qid")
            .agg(
                # reciprocal rank of the first relevant item in top-k (1/rank); 0 if none is relevant in top-k
                value=pl.when(relevant_in_top_k.any()).then(1.0 / (relevant_in_top_k.arg_max() + 1)).otherwise(0.0),
            )
            .select("value")
            .mean()
            .collect()
            .item()
        )


class PrecisionAtK(AtKMetric):
    def calculate(self, samples: pl.DataFrame, _artifacts: Artifacts, /) -> float:
        return (
            samples.lazy()
            .group_by("qid")
            .agg(
                value=(pl.col("relevance").sort_by("score", descending=True).head(self.k) > 0).cast(pl.Float32()).sum()
                / self.k,
            )
            .select("value")
            .mean()
            .collect()
            .item()
        )


class RecallAtK(AtKMetric):
    def calculate(self, samples: pl.DataFrame, _artifacts: Artifacts, /) -> float:
        return (
            samples.lazy()
            .group_by("qid")
            .agg(
                value=pl.when(pl.col("relevance").max() > 0)
                .then(
                    (pl.col("relevance").sort_by("score", descending=True).head(self.k) > 0).cast(pl.Float32()).sum()
                    / (pl.col("relevance") > 0).cast(pl.Float32()).sum(),
                )
                .otherwise(0.0),
            )
            .select("value")
            .mean()
            .collect()
            .item()
        )


class NdcgAtK(AtKMetric):
    def calculate(self, samples: pl.DataFrame, _artifacts: Artifacts, /) -> float:
        return (
            samples.lazy()
            .group_by("qid")
            .agg(
                value=pl.when(pl.col("relevance").max() > 0)
                .then(
                    (
                        pl.col("relevance").sort_by("score", descending=True).head(self.k).cast(pl.Float32())
                        / pl.col("relevance").head(self.k).cum_count().add(1).log(2).cast(pl.Float32())
                    ).sum()
                    / (
                        pl.col("relevance").sort_by("relevance", descending=True).head(self.k).cast(pl.Float32())
                        / pl.col("relevance").head(self.k).cum_count().add(1).log(2).cast(pl.Float32())
                    ).sum(),
                )
                .otherwise(0.0),
            )
            .select("value")
            .mean()
            .collect()
            .item()
        )


class ScoreShiftAucAtK(AtKMetric):
    """Measures the smoothness of score changes in the top k recommendations.

    Calculates the area under the normalized score curve (AUC) from rank 1 to recs len, averaged over all clients.
    Scores are normalized for each client separately using min-max scaling.

    Interpretation:
    * 0.0 - Sharp drop in scores (model is heavily biased towards top-1)
    * 0.5 - Linear decrease in scores (natural distribution)
    * 1.0 - Perfectly uniform distribution (all scores are the same)
    """

    def calculate(self, samples: pl.DataFrame, _artifacts: Artifacts, /) -> float:
        top_k = samples.lazy().group_by("qid").agg(pl.col("score").top_k(self.k).sort(descending=True))

        return (
            (
                top_k.with_columns(
                    pl.col("score").list.eval(
                        pl.when(pl.element().max() != pl.element().min())
                        .then((pl.element() - pl.element().min()) / (pl.element().max() - pl.element().min()))
                        .otherwise(1.0),
                    ),
                )
                .with_columns(
                    pl.col("score")
                    .list.eval(
                        pl.when(pl.element().len() > 1)
                        .then(
                            ((pl.element() + pl.element().shift(-1)).drop_nulls().sum() / 2) / (pl.element().len() - 1),
                        )
                        .otherwise(1.0),
                    )
                    .list.first(),
                )
                .select(pl.mean("score"))
            )
            .collect()
            .item()
        )


class InterlistDiversityAtK(AtKMetric):
    """Measures the average overlap between recommendation lists of different clients.

    Calculates the average intersection size divided by min len recs for all user pairs (except self),
    then returns one minus that value to get diversity score.

    Interpretation:
    * 0.0 - All users receive identical recommendations (full overlap)
    * 0.5 - Half of items overlap between user lists on average
    * 1.0 - All users receive completely different recommendations (no overlap)

    NB: Works really slow and requires a lot of RAM. Please use a subset of <= 1000 clients.
    """

    def calculate(self, samples: pl.DataFrame, _artifacts: Artifacts, /) -> float:
        top_k = samples.lazy().group_by("qid").agg(items=pl.col("item").top_k_by("score", self.k)).with_row_index("i")

        return (
            (
                top_k.join(
                    top_k.select(
                        pl.col("i").alias("j"),
                        pl.col("items").alias("items_j"),
                    ),
                    how="cross",
                )
                .filter(pl.col("i") != pl.col("j"))
                .with_columns(
                    intersection=pl.col("items").list.set_intersection(pl.col("items_j")).list.len(),
                    min_len=pl.min_horizontal(
                        pl.col("items").list.len(),
                        pl.col("items_j").list.len(),
                    ),
                )
                .with_columns(
                    overlap_ratio=pl.when(pl.col("min_len") > 0)
                    .then(pl.col("intersection") / pl.col("min_len"))
                    .otherwise(0.0),
                )
                .select(diversity_score=(1 - pl.col("overlap_ratio").mean()))
            )
            .collect()
            .item()
        )


class CoverageAtK(AtKMetric):
    """Measures the proportion of unique items (or categories) appearing in top-k recommendations.

    Calculates the ratio of distinct items recommended at least min_occurrence times
    to the total number of unique items in the dataset.

    Interpretation:
    * 0.0 - No items are recommended (or all below min_occurrence threshold)
    * 1.0 - All items in the dataset appear in recommendations
    """

    def __init__(self, *, k: int, feature: str = "item", min_occurrence: int = 1) -> None:
        super().__init__(k=k)

        self.feature = feature
        self.min_occurrence = min_occurrence

    def calculate(self, samples: pl.DataFrame, artifacts: Artifacts, /) -> float:
        top_k = samples.lazy().sort("score", descending=True).group_by("qid").head(self.k).select("item")

        if self.feature != "item":
            top_k = (
                top_k.join(
                    artifacts.items.lazy().select("item", self.feature),
                    on="item",
                    how="inner",  # only counting for items that have feature
                    maintain_order="left",
                )
                .select(self.feature)
                .drop_nulls()
            )

        return (
            (top_k.group_by(self.feature).len().filter(pl.col("len") >= self.min_occurrence)).select(pl.len()).collect()
            / artifacts.items.select(self.feature).n_unique()
        ).item()


class MaxStreakAtK(AtKMetric):
    """Measures the maximum length of consecutive equal values in top-k recommendations.

    Calculates the longest streak of items with the same value in the specified feature,
    averaged over all clients.

    Interpretation:
    * 1.0 - No streaks (all consecutive items have different values)
    * Higher values - Longer sequences with equal values
    * k - All items have the same value in entire top-k
    """

    def __init__(self, *, k: int, feature: str) -> None:
        super().__init__(k=k)

        self.feature = feature

    def calculate(self, samples: pl.DataFrame, artifacts: Artifacts, /) -> float:
        top_k = (
            samples.lazy()
            .sort("score", descending=True)
            .group_by("qid")
            .head(self.k)
            .join(
                artifacts.items.lazy().select("item", self.feature),
                on="item",
                how="left",
                maintain_order="left",
            )
        )

        return (
            (
                top_k.group_by("qid")
                .agg(pl.col(self.feature))
                .select(
                    max_streak=pl.col(self.feature)
                    .list.eval(pl.element().rle().struct.field("len").max())
                    .list.first(),
                )
                .mean()
            )
            .collect()
            .item()
        )


class IntralistSimilarityAtK(AtKMetric):
    """Measures the average pairwise cosine similarity between items within each recommendation list.

    Interpretation:
    * +1 - All items in all lists are identical in embedding space (are similar)
    * 0 - Items within lists are orthogonal/unrelated on average (are dissimilar)
    * -1 - Items within lists point in opposite directions (are opposites of each other whatever it means)
    Usually the results are somewhere from 0 to 1.

    NB: Works really slow and requires a lot of RAM. Please use a subset of <= 100_000 clients.
    """

    def __init__(self, *, k: int, embedding_feature: str = "embedding") -> None:
        super().__init__(k=k)

        self.k = k
        self.embedding_feature = embedding_feature  # must contain normalized embeddings

    def _check_embedding_feature_dtype(self, artifacts: Artifacts) -> None:
        dtype = artifacts.items.schema[self.embedding_feature]
        if not isinstance(dtype, (pl.List, pl.Array)):
            raise TypeError(
                f"Expected '{self.embedding_feature}' to be List or Array, but got {dtype}. "
                "Embeddings must be array-like (list or fixed-size array).",
            )

    def _pairwise_cosine_similarity(self, embeddings: list[np.ndarray]) -> float:
        if len(embeddings) <= 1:
            return 0.0
        emb_matrix = np.vstack(embeddings)
        similarity_matrix = emb_matrix @ emb_matrix.T
        similarities = similarity_matrix[np.triu_indices(len(embeddings), k=1)]
        return float(similarities.mean())

    def calculate(self, samples: pl.DataFrame, artifacts: Artifacts, /) -> float:
        self._check_embedding_feature_dtype(artifacts)

        top_k = (
            samples.lazy()
            .sort("score", descending=True)
            .group_by("qid")
            .head(self.k)
            .join(
                artifacts.items.lazy().select("item", self.embedding_feature),
                on="item",
                how="inner",  # only counting for items that have embeddings
            )
        )

        return (
            (
                top_k.group_by("qid")
                .agg(pl.col(self.embedding_feature))
                .filter(pl.col(self.embedding_feature).list.len() > 1)
                .select(
                    pl.col(self.embedding_feature)
                    .map_batches(
                        lambda s: pl.Series(
                            [self._pairwise_cosine_similarity(embeddings) for embeddings in s],
                            dtype=pl.Float32,
                        ),
                        return_dtype=pl.Float32,
                    )
                    .mean(),
                )
            )
            .collect()
            .item()
        )


class EntropyAtK(AtKMetric):
    """Measures the diversity using Shannon entropy, normalized to [0, 1] range by maximum possible entropy.

    Interpretation:
    * 0.0 - All items have identical values (minimum diversity)
    * 1.0 - Perfectly uniform distribution of values (maximum diversity)
    """

    def __init__(self, *, k: int, feature: str) -> None:
        super().__init__(k=k)

        self.feature = feature

    def calculate(self, samples: pl.DataFrame, artifacts: Artifacts, /) -> float:
        top_k = (
            samples.lazy()
            .sort("score", descending=True)
            .group_by("qid")
            .head(self.k)
            .join(
                artifacts.items.lazy().select("item", self.feature),
                on="item",
                how="left",
            )
        )

        return (
            (
                top_k.group_by("qid")
                .agg(
                    proportions=pl.col(self.feature).value_counts(normalize=True).struct.field("proportion"),
                )
                .with_columns(
                    pl.when(pl.col("proportions").list.len() > 1)
                    .then(
                        pl.col("proportions")
                        .list.eval(
                            pl.element().log(2) * pl.element(),
                        )
                        .list.sum()
                        * -1
                        / pl.col("proportions").list.len().log(2),
                    )
                    .otherwise(0.0),
                )
                .select(pl.col("proportions").mean())
            )
            .collect()
            .item()
        )


class GiniByPopularityAtK(AtKMetric):
    """Measures the inequality in item recommendation frequency using the Gini coefficient.

    Interpretation:
    * 0.0 - Perfect inequality (few items get all recommendations or the long tail issue)
    * 1.0 - Perfect equality (all items recommended equally)

    This metric considers only top-k recommendations for each client
    and only for those items that were recommended with the frequency of at least min_occurrence_share.
    """

    def __init__(self, *, k: int, min_occurrence_share: float = 0.0) -> None:
        super().__init__(k=k)

        self.min_occurrence_share = min_occurrence_share

    def calculate(self, samples: pl.DataFrame, artifacts: Artifacts, /) -> float:
        n_queries = samples.select("qid").n_unique()

        top_k = samples.lazy().sort("score", descending=True).group_by("qid").head(self.k)

        return (
            (
                top_k.group_by("item")
                .agg(count=pl.len())
                .with_columns(popularity=pl.col("count") / n_queries)
                .join(
                    artifacts.items.lazy().select("item"),
                    on="item",
                    how="right",
                )
                .fill_null(0.0)
                .filter(pl.col("popularity") >= self.min_occurrence_share)
                .sort("popularity")
                .select(
                    pl.when(pl.col("popularity").len() == 0)
                    .then(0.0)
                    .otherwise(
                        1.0
                        - (
                            (pl.col("popularity").len() + 1) / pl.col("popularity").len()
                            - (
                                2
                                * pl.col("popularity").dot(
                                    pl.col("popularity").len() + 1 - pl.int_range(1, pl.col("popularity").len() + 1),
                                )
                                / (pl.col("popularity").len() * pl.col("popularity").sum())
                            )
                        ),
                    ),
                )
            )
            .collect()
            .item()
        )
