from perseus.core.tasks import base
from perseus.core.tasks.classification.evaluation import F1, Accuracy, Evaluator, PrAuc, Precision, Recall, RocAuc
from perseus.core.tasks.classification.target import Artifacts, Criterion, Head, Layer, Preprocessor

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
        "recall": Recall,
        "precision": Precision,
        "f1": F1,
        "roc_auc": RocAuc,
        "pr_auc": PrAuc,
    },
)
