import inspect
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
from perseus.core.tasks.classification.target import Artifacts


class Evaluator(base.Evaluator):
    @property
    def predict_kwargs(self) -> dict[str, t.Any]:
        return {}

    def _prepare_samples(self, samples: pl.DataFrame, _artifacts: Artifacts, /) -> pl.DataFrame:
        return samples.with_columns(
            most_probable_label=pl.concat_list(pl.col("prediction").struct.field("*"))
            .list.arg_max()
            .replace_strict(dict(enumerate(field.name for field in samples.schema["prediction"].fields))),
        )


class SklearnScorer(t.Protocol):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs: t.Any) -> float: ...


class SklearnMetric(base.Metric):
    _score_fn: t.ClassVar[SklearnScorer]

    def __init__(self, threshold: float | None = None, **kwargs: t.Any) -> None:
        self.threshold = threshold
        self.kwargs = kwargs

    def calculate(self, samples: pl.DataFrame, _artifacts: Artifacts, /) -> float:
        score_fn = type(self)._score_fn  # noqa: SLF001

        if self.threshold is not None:
            if samples["target"].n_unique() != 2:
                raise ValueError("threshold can be used only in binary classification case")
            if "pos_label" not in self.kwargs:
                raise ValueError("`pos_label` must be set in binary classification case to use `threshold`")
            pos_label = self.kwargs["pos_label"]
            y_true = (samples["target"] == pos_label).to_numpy()
            y_pred = (samples["prediction"].struct.field(pos_label) > self.threshold).to_numpy()
            kwargs = self.kwargs.copy()
            if "pos_label" in inspect.signature(score_fn).parameters:
                kwargs["pos_label"] = True
            else:
                kwargs.pop("pos_label")
        else:
            y_true = samples["target"].to_numpy()
            y_pred = samples["most_probable_label"].to_numpy()
            kwargs = self.kwargs

        return score_fn(y_true, y_pred, **kwargs)


class Accuracy(SklearnMetric):
    _score_fn = accuracy_score


class Recall(SklearnMetric):
    _score_fn = recall_score


class Precision(SklearnMetric):
    _score_fn = precision_score


class F1(SklearnMetric):
    _score_fn = f1_score


class RocAuc(base.Metric):
    def __init__(self, *, pos_label: str | None = None, **kwargs: t.Any) -> None:
        super().__init__()

        self.pos_label = pos_label
        self.kwargs = kwargs

    def calculate(self, samples: pl.DataFrame, _artifacts: Artifacts, /) -> float:
        if samples["target"].n_unique() == 2:
            if self.pos_label is None:
                raise ValueError("in binary classification case pos_label must be set")
            y_true = (samples["target"] == self.pos_label).to_numpy()
            y_pred = samples["prediction"].struct.field(self.pos_label).to_numpy()
            labels = None
        else:
            y_true = samples["target"].to_numpy()
            y_pred = samples["prediction"].struct.unnest().to_numpy()
            labels = [field.name for field in samples.schema["prediction"].fields]

        kwargs = self.kwargs.copy()
        kwargs.setdefault("labels", labels)
        return roc_auc_score(y_true, y_pred, **kwargs)


class PrAuc(base.Metric):
    def __init__(self, *, pos_label: str | None = None, **kwargs: t.Any) -> None:
        super().__init__()

        self.pos_label = pos_label
        self.kwargs = kwargs

    def calculate(self, samples: pl.DataFrame, _artifacts: Artifacts, /) -> float:
        if samples["target"].n_unique() == 2:
            if self.pos_label is None:
                raise ValueError("in binary classification case pos_label must be set")
            y_true = (samples["target"] == self.pos_label).to_numpy()
            y_pred = samples["prediction"].struct.field(self.pos_label).to_numpy()
            pos_label = True
        else:
            y_true = samples["target"].to_numpy()
            y_pred = samples["prediction"].struct.unnest().to_numpy()
            pos_label = 1

        return average_precision_score(y_true, y_pred, pos_label=pos_label, **self.kwargs)
