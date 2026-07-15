from perseus.core.tasks import base
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
from perseus.core.tasks.retrieval.target import Artifacts, Criterion, Head, Layer, Preprocessor

__all__ = [
    "task",
]

task = base.Task(
    Artifacts,
    Preprocessor,
    Layer,
    Criterion,
    Head,
    Evaluator,
    {
        "recall_at_k": RecallAtK,
        "precision_at_k": PrecisionAtK,
        "hit_rate_at_k": HitRateAtK,
        "mrr_at_k": MrrAtK,
        "ndcg_at_k": NdcgAtK,
        "coverage_at_k": CoverageAtK,
        "max_streak_at_k": MaxStreakAtK,
        "intralist_similarity_at_k": IntralistSimilarityAtK,
        "entropy_at_k": EntropyAtK,
        "gini_by_popularity_at_k": GiniByPopularityAtK,
        "score_shift_auc_at_k": ScoreShiftAucAtK,
        "interlist_diversity_at_k": InterlistDiversityAtK,
    },
)
