from dataclasses import dataclass

from perseus.core.tasks.base.evaluation import Evaluator, Metric
from perseus.core.tasks.base.target import Artifacts, Criterion, Head, Layer, Preprocessor

__all__ = [
    "Artifacts",
    "Evaluator",
    "Head",
    "Metric",
    "Preprocessor",
    "Task",
]


@dataclass(frozen=True)
class Task:
    artifacts: type[Artifacts]
    preprocessor: type[Preprocessor]
    layer: type[Layer]
    criterion: Criterion
    head: type[Head]
    evaluator: type[Evaluator]
    metrics: dict[str, type[Metric]]
