from perseus.core.tasks import base
from perseus.core.tasks.ranking.evaluation import (
    F1,
    Accuracy,
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
    PrAuc,
    Precision,
    PrecisionAtK,
    Recall,
    RecallAtK,
    RocAuc,
    ScoreShiftAucAtK,
)
from perseus.core.tasks.ranking.target import Artifacts, Criterion, Head, Layer, Preprocessor

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
        "accuracy": Accuracy,
        "precision": Precision,
        "recall": Recall,
        "f1": F1,
        "roc_auc": RocAuc,
        "pr_auc": PrAuc,
        "coverage_at_k": CoverageAtK,
        "max_streak_at_k": MaxStreakAtK,
        "intralist_similarity_at_k": IntralistSimilarityAtK,
        "entropy_at_k": EntropyAtK,
        "gini_by_popularity_at_k": GiniByPopularityAtK,
        "hit_rate_at_k": HitRateAtK,
        "mrr_at_k": MrrAtK,
        "precision_at_k": PrecisionAtK,
        "recall_at_k": RecallAtK,
        "ndcg_at_k": NdcgAtK,
        "score_shift_auc_at_k": ScoreShiftAucAtK,
        "interlist_diversity_at_k": InterlistDiversityAtK,
    },
)
