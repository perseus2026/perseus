"""Tests for ranking metrics.

The metrics operate on "prepared" samples with qid/item/relevance/score columns
(the result of Evaluator._prepare_samples). Here they are tested directly, and artifacts
is replaced with a lightweight object holding an .items attribute.
"""

import types

import polars as pl
import pytest

from perseus.core.tasks.ranking.evaluation import (
    F1,
    Accuracy,
    CoverageAtK,
    EntropyAtK,
    GiniByPopularityAtK,
    HitRateAtK,
    InterlistDiversityAtK,
    IntralistSimilarityAtK,
    MaxStreakAtK,
    MrrAtK,
    NdcgAtK,
    PrAuc,
    Precision,
    PrecisionAtK,
    Recall,
    RecallAtK,
    RocAuc,
    ScoreShiftAucAtK,
)


def _artifacts(items: pl.DataFrame) -> types.SimpleNamespace:
    return types.SimpleNamespace(items=items)


@pytest.fixture
def ranked() -> pl.DataFrame:
    # qid0: A(rel1,0.9) > B(rel0,0.5) > C(rel0,0.2)
    # qid1: D(rel0,0.8) > E(rel2,0.3)
    return pl.DataFrame(
        {
            "qid": [0, 0, 0, 1, 1],
            "item": ["A", "B", "C", "D", "E"],
            "relevance": [1, 0, 0, 0, 2],
            "score": [0.9, 0.5, 0.2, 0.8, 0.3],
        },
    )


def test_hit_rate_at_1(ranked: pl.DataFrame) -> None:
    # qid0 top1=A(rel1)->hit; qid1 top1=D(rel0)->miss
    assert HitRateAtK(k=1).calculate(ranked, None) == pytest.approx(0.5)


def test_hit_rate_at_2(ranked: pl.DataFrame) -> None:
    assert HitRateAtK(k=2).calculate(ranked, None) == pytest.approx(1.0)


def test_mrr_is_mean_reciprocal_rank(ranked: pl.DataFrame) -> None:
    # qid0: first relevant item at position 1 -> 1/1; qid1: at position 2 -> 1/2.
    # MRR = (1.0 + 0.5) / 2 = 0.75
    assert MrrAtK(k=3).calculate(ranked, None) == pytest.approx(0.75)


def test_mrr_zero_when_relevant_outside_top_k() -> None:
    # a relevant item exists but is outside top-1 -> contributes 0
    samples = pl.DataFrame(
        {
            "qid": [0, 0],
            "item": ["A", "B"],
            "relevance": [0, 1],
            "score": [0.9, 0.1],
        },
    )
    assert MrrAtK(k=1).calculate(samples, None) == pytest.approx(0.0)


def test_precision_at_k(ranked: pl.DataFrame) -> None:
    # qid0: top2=[A(rel1),B(rel0)] -> 1 relevant /2 =0.5; qid1: top2=[D(0),E(2)] ->1/2=0.5
    assert PrecisionAtK(k=2).calculate(ranked, None) == pytest.approx(0.5)


def test_recall_at_k(ranked: pl.DataFrame) -> None:
    # qid0: 1 relevant found out of 1 -> 1.0; qid1: 1 of 1 -> 1.0
    assert RecallAtK(k=2).calculate(ranked, None) == pytest.approx(1.0)


def test_ndcg_perfect_ranking_is_one() -> None:
    # order by score matches order by relevance -> NDCG = 1.0
    samples = pl.DataFrame(
        {
            "qid": [0, 0, 0],
            "item": ["A", "B", "C"],
            "relevance": [3, 2, 1],
            "score": [0.9, 0.5, 0.1],
        },
    )
    assert NdcgAtK(k=3).calculate(samples, None) == pytest.approx(1.0)


def test_ndcg_imperfect_in_unit_range(ranked: pl.DataFrame) -> None:
    value = NdcgAtK(k=3).calculate(ranked, None)
    assert 0.0 < value < 1.0


def test_score_shift_uniform_scores_is_one() -> None:
    samples = pl.DataFrame(
        {
            "qid": [0, 0, 0],
            "item": ["A", "B", "C"],
            "relevance": [0, 0, 0],
            "score": [0.5, 0.5, 0.5],
        },
    )
    assert ScoreShiftAucAtK(k=3).calculate(samples, None) == pytest.approx(1.0)


def test_interlist_diversity_identical_lists_is_zero() -> None:
    samples = pl.DataFrame(
        {
            "qid": [0, 0, 1, 1],
            "item": ["A", "B", "A", "B"],
            "relevance": [0, 0, 0, 0],
            "score": [0.9, 0.5, 0.9, 0.5],
        },
    )
    assert InterlistDiversityAtK(k=2).calculate(samples, None) == pytest.approx(0.0)


def test_interlist_diversity_disjoint_lists_is_one() -> None:
    samples = pl.DataFrame(
        {
            "qid": [0, 0, 1, 1],
            "item": ["A", "B", "C", "D"],
            "relevance": [0, 0, 0, 0],
            "score": [0.9, 0.5, 0.9, 0.5],
        },
    )
    assert InterlistDiversityAtK(k=2).calculate(samples, None) == pytest.approx(1.0)


def test_coverage_at_k(ranked: pl.DataFrame) -> None:
    samples = pl.DataFrame(
        {
            "qid": [0, 0, 1, 1],
            "item": ["A", "B", "A", "C"],
            "relevance": [0, 0, 0, 0],
            "score": [0.9, 0.5, 0.8, 0.3],
        },
    )
    artifacts = _artifacts(pl.DataFrame({"item": ["A", "B", "C", "D"]}))
    # recommended distinct {A,B,C}=3 out of 4 unique -> 0.75
    assert CoverageAtK(k=2).calculate(samples, artifacts) == pytest.approx(0.75)


def test_max_streak_at_k() -> None:
    samples = pl.DataFrame(
        {
            "qid": [0, 0, 1, 1],
            "item": ["A", "B", "A", "C"],
            "relevance": [0, 0, 0, 0],
            "score": [0.9, 0.5, 0.8, 0.3],
        },
    )
    # A,B -> cat [x,x] streak 2; A,C -> cat [x,y] streak 1; mean = 1.5
    artifacts = _artifacts(pl.DataFrame({"item": ["A", "B", "C"], "cat": ["x", "x", "y"]}))
    assert MaxStreakAtK(k=2, feature="cat").calculate(samples, artifacts) == pytest.approx(1.5)


def test_entropy_at_k() -> None:
    samples = pl.DataFrame(
        {
            "qid": [0, 0, 1, 1],
            "item": ["A", "B", "A", "C"],
            "relevance": [0, 0, 0, 0],
            "score": [0.9, 0.5, 0.8, 0.3],
        },
    )
    # qid0 cats [x,x] -> entropy 0; qid1 cats [x,y] -> entropy 1; mean = 0.5
    artifacts = _artifacts(pl.DataFrame({"item": ["A", "B", "C"], "cat": ["x", "x", "y"]}))
    assert EntropyAtK(k=2, feature="cat").calculate(samples, artifacts) == pytest.approx(0.5)


def test_intralist_similarity_identical_embeddings_is_one() -> None:
    samples = pl.DataFrame(
        {
            "qid": [0, 0],
            "item": ["A", "B"],
            "relevance": [0, 0],
            "score": [0.9, 0.5],
        },
    )
    artifacts = _artifacts(
        pl.DataFrame({"item": ["A", "B"], "embedding": [[1.0, 0.0], [1.0, 0.0]]}),
    )
    assert IntralistSimilarityAtK(k=2).calculate(samples, artifacts) == pytest.approx(1.0)


def test_intralist_similarity_rejects_non_array_feature() -> None:
    samples = pl.DataFrame({"qid": [0], "item": ["A"], "relevance": [0], "score": [0.9]})
    artifacts = _artifacts(pl.DataFrame({"item": ["A"], "embedding": [1.0]}))
    with pytest.raises(TypeError, match="List or Array"):
        IntralistSimilarityAtK(k=2).calculate(samples, artifacts)


def test_gini_in_unit_range() -> None:
    samples = pl.DataFrame(
        {
            "qid": [0, 0, 1, 1],
            "item": ["A", "B", "A", "C"],
            "relevance": [0, 0, 0, 0],
            "score": [0.9, 0.5, 0.8, 0.3],
        },
    )
    artifacts = _artifacts(pl.DataFrame({"item": ["A", "B", "C", "D"]}))
    value = GiniByPopularityAtK(k=2).calculate(samples, artifacts)
    assert 0.0 <= value <= 1.0


@pytest.fixture
def binary_ranked() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "click": [1, 0, 1, 0],
            "click_prediction": [0.9, 0.2, 0.4, 0.6],
        },
    )


def test_sklearn_metrics(binary_ranked: pl.DataFrame) -> None:
    # threshold 0.5 -> pred [T,F,F,T]; true [1,0,1,0] -> accuracy 2/4 = 0.5
    assert Accuracy(label="click").calculate(binary_ranked, None) == pytest.approx(0.5)
    assert 0.0 <= Precision(label="click").calculate(binary_ranked, None) <= 1.0
    assert 0.0 <= Recall(label="click").calculate(binary_ranked, None) <= 1.0
    assert 0.0 <= F1(label="click").calculate(binary_ranked, None) <= 1.0


def test_roc_auc(binary_ranked: pl.DataFrame) -> None:
    assert 0.0 <= RocAuc(label="click").calculate(binary_ranked, None) <= 1.0


def test_pr_auc(binary_ranked: pl.DataFrame) -> None:
    assert 0.0 <= PrAuc(label="click").calculate(binary_ranked, None) <= 1.0
