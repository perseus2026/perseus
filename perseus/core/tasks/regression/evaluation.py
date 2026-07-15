import typing as t

import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from perseus.core.tasks import base
from perseus.core.tasks.regression.target import Artifacts


class Evaluator(base.Evaluator):
    @property
    def predict_kwargs(self) -> dict[str, t.Any]:
        return {}

    def _prepare_samples(self, samples: pl.DataFrame, _artifacts: Artifacts) -> pl.DataFrame:
        return samples


class SklearnScorer(t.Protocol):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs: t.Any) -> float: ...


class SklearnMetric(base.Metric):
    _score_fn: t.ClassVar[SklearnScorer]

    def __init__(self, **kwargs: t.Any) -> None:
        self.kwargs = kwargs

    def calculate(self, samples: pl.DataFrame, _artifacts: Artifacts, /) -> float:
        score_fn = type(self)._score_fn  # noqa: SLF001
        y_true = samples["target"].to_numpy()
        y_pred = samples["prediction"].to_numpy()
        return score_fn(y_true, y_pred, **self.kwargs)


class Rmse(SklearnMetric):
    _score_fn = root_mean_squared_error


class Mae(SklearnMetric):
    _score_fn = mean_absolute_error


class R2(SklearnMetric):
    _score_fn = r2_score
