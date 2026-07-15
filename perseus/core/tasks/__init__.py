from perseus.core.tasks import classification, ranking, regression, retrieval
from perseus.core.tasks.base import (
    Artifacts,
    Criterion,
    Evaluator,
    Head,
    Layer,
    Metric,
    Preprocessor,
    Task,
)

__all__ = [
    "Artifacts",
    "Criterion",
    "Evaluator",
    "Head",
    "Layer",
    "Metric",
    "Observer",
    "Preprocessor",
    "Task",
]

registry: dict[str, Task] = {
    "regression": regression.task,
    "classification": classification.task,
    "retrieval": retrieval.task,
    "ranking": ranking.task,
}
