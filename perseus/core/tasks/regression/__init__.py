from perseus.core.tasks import base
from perseus.core.tasks.regression.evaluation import R2, Evaluator, Mae, Rmse
from perseus.core.tasks.regression.target import Artifacts, Criterion, Head, Layer, Preprocessor

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
        "rmse": Rmse,
        "mae": Mae,
        "r2": R2,
    },
)
