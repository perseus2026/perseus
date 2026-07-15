"""Тесты метрик retrieval.

Часть метрик (HitRate/Mrr/Precision/Recall/Ndcg) работает на «подготовленных» samples
с колонками-списками prediction_relevances/target_relevances. Остальные — на сыром
prediction (список структур {item, score}). artifacts подменяется на объект с .items.
"""

import types

import polars as pl
import pytest

from perseus.core.tasks.retrieval.evaluation import (
    CoverageAtK,
    EntropyAtK,
    Evaluator,
    GiniByPopularityAtK,
    HitRateAtK,
    InterlistDiversityAtK,
    IntralistSimilarityAtK,
    MaxStreakAtK,
    MrrAtK,
    NdcgAtK,
    PrecisionAtK,
    RecallAtK,
    ScoreShiftAucAtK,
)


def _artifacts(items: pl.DataFrame) -> types.SimpleNamespace:
    return types.SimpleNamespace(items=items)


@pytest.fixture
def prepared() -> pl.DataFrame:
    # q0: предсказанные релевантности [1,0,0], целевые [1]; q1: [0,2] и [2]
    return pl.DataFrame(
        {
            "prediction_relevances": [[1.0, 0.0, 0.0], [0.0, 2.0]],
            "target_relevances": [[1.0], [2.0]],
        },
    )


def _predictions(rows: list[list[tuple[str, float]]]) -> pl.DataFrame:
    return pl.DataFrame(
        {"prediction": [[{"item": item, "score": score} for item, score in row] for row in rows]},
    )


# --- метрики на prediction_relevances ---


def test_hit_rate(prepared: pl.DataFrame) -> None:
    assert HitRateAtK(k=1).calculate(prepared, None) == pytest.approx(0.5)
    assert HitRateAtK(k=2).calculate(prepared, None) == pytest.approx(1.0)


def test_mrr_is_reciprocal_rank(prepared: pl.DataFrame) -> None:
    # q0 первый релевантный на позиции 1 → 1/1; q1 на позиции 2 → 1/2 → mean 0.75
    assert MrrAtK(k=3).calculate(prepared, None) == pytest.approx(0.75)


def test_precision(prepared: pl.DataFrame) -> None:
    # q0: 1 релевантный из top2 /2 = 0.5; q1: 1/2 = 0.5
    assert PrecisionAtK(k=2).calculate(prepared, None) == pytest.approx(0.5)


def test_recall(prepared: pl.DataFrame) -> None:
    # каждый запрос находит свой единственный релевантный → 1.0
    assert RecallAtK(k=2).calculate(prepared, None) == pytest.approx(1.0)


def test_ndcg_perfect_is_one() -> None:
    samples = pl.DataFrame(
        {
            "prediction_relevances": [[3.0, 2.0, 1.0]],
            "target_relevances": [[3.0, 2.0, 1.0]],
        },
    )
    assert NdcgAtK(k=3).calculate(samples, None) == pytest.approx(1.0)


def test_ndcg_imperfect_in_unit_range(prepared: pl.DataFrame) -> None:
    value = NdcgAtK(k=3).calculate(prepared, None)
    assert 0.0 < value <= 1.0


# --- _prepare_samples и Evaluator ---


def test_prepare_samples_builds_relevance_lists() -> None:
    samples = pl.DataFrame(
        {
            "target": [[{"item": "A", "relevance": 1}, {"item": "B", "relevance": 2}]],
            "prediction": [[{"item": "B", "score": 0.9}, {"item": "A", "score": 0.5}, {"item": "C", "score": 0.1}]],
        },
    )
    out = Evaluator({})._prepare_samples(samples, None)
    # prediction_relevances: релевантности в порядке убывания score → B(2),A(1),C(0)
    assert out["prediction_relevances"].to_list() == [[2, 1, 0]]
    assert out["target_relevances"].to_list() == [[1, 2]]


def test_evaluator_predict_kwargs_uses_max_k() -> None:
    evaluator = Evaluator({"hr": HitRateAtK(k=5), "mrr": MrrAtK(k=10)})
    assert evaluator.predict_kwargs == {"k": 10}


# --- метрики на сыром prediction + artifacts ---


def test_coverage() -> None:
    samples = _predictions([[("A", 0.9), ("B", 0.5)], [("A", 0.8), ("C", 0.3)]])
    artifacts = _artifacts(pl.DataFrame({"item": ["A", "B", "C", "D"]}))
    # рекомендованы distinct {A,B,C}=3 из 4 → 0.75
    assert CoverageAtK(k=2).calculate(samples, artifacts) == pytest.approx(0.75)


def test_max_streak() -> None:
    samples = _predictions([[("A", 0.9), ("B", 0.5)], [("A", 0.8), ("C", 0.3)]])
    artifacts = _artifacts(pl.DataFrame({"item": ["A", "B", "C"], "cat": ["x", "x", "y"]}))
    # q0 cats [x,x] streak 2; q1 [x,y] streak 1 → mean 1.5
    assert MaxStreakAtK(k=2, feature="cat").calculate(samples, artifacts) == pytest.approx(1.5)


def test_entropy() -> None:
    samples = _predictions([[("A", 0.9), ("B", 0.5)], [("A", 0.8), ("C", 0.3)]])
    artifacts = _artifacts(pl.DataFrame({"item": ["A", "B", "C"], "cat": ["x", "x", "y"]}))
    # q0 энтропия 0, q1 энтропия 1 → mean 0.5
    assert EntropyAtK(k=2, feature="cat").calculate(samples, artifacts) == pytest.approx(0.5)


def test_gini_in_unit_range() -> None:
    samples = _predictions([[("A", 0.9), ("B", 0.5)], [("A", 0.8), ("C", 0.3)]])
    artifacts = _artifacts(pl.DataFrame({"item": ["A", "B", "C", "D"]}))
    value = GiniByPopularityAtK(k=2).calculate(samples, artifacts)
    assert 0.0 <= value <= 1.0


def test_score_shift_uniform_is_one() -> None:
    samples = _predictions([[("A", 0.5), ("B", 0.5), ("C", 0.5)]])
    assert ScoreShiftAucAtK(k=3).calculate(samples, None) == pytest.approx(1.0)


def test_interlist_diversity_identical_is_zero() -> None:
    samples = _predictions([[("A", 0.9), ("B", 0.5)], [("A", 0.9), ("B", 0.5)]])
    assert InterlistDiversityAtK(k=2).calculate(samples, None) == pytest.approx(0.0)


def test_interlist_diversity_disjoint_is_one() -> None:
    samples = _predictions([[("A", 0.9), ("B", 0.5)], [("C", 0.9), ("D", 0.5)]])
    assert InterlistDiversityAtK(k=2).calculate(samples, None) == pytest.approx(1.0)


def test_intralist_similarity_identical_embeddings_is_one() -> None:
    samples = _predictions([[("A", 0.9), ("B", 0.5)]])
    artifacts = _artifacts(pl.DataFrame({"item": ["A", "B"], "embedding": [[1.0, 0.0], [1.0, 0.0]]}))
    assert IntralistSimilarityAtK(k=2).calculate(samples, artifacts) == pytest.approx(1.0)


def test_intralist_similarity_rejects_non_array_feature() -> None:
    samples = _predictions([[("A", 0.9)]])
    artifacts = _artifacts(pl.DataFrame({"item": ["A"], "embedding": [1.0]}))
    with pytest.raises(TypeError, match="List or Array"):
        IntralistSimilarityAtK(k=2).calculate(samples, artifacts)
