import typing as t
from abc import ABC, abstractmethod
from collections import defaultdict

import polars as pl

from perseus.core.tasks.base.target import Artifacts


class Metric(ABC):
    @abstractmethod
    def calculate(self, samples: pl.DataFrame, artifacts: Artifacts, /) -> float: ...


class Evaluator(ABC):
    def __init__(self, name_to_metric: dict[str, Metric], /) -> None:
        self.name_to_metric = name_to_metric

    @property
    @abstractmethod
    def predict_kwargs(self) -> dict[str, t.Any]: ...

    def calculate_metrics(self, samples: pl.DataFrame, artifacts: Artifacts, /) -> dict[str, dict[str, float]]:
        samples = self._prepare_samples(samples, artifacts)

        metric_to_group_to_value = defaultdict(dict)
        for name, metric in self.name_to_metric.items():
            metric_to_group_to_value[name]["overall"] = metric.calculate(samples, artifacts)

        if (groups_dtype := samples.schema.get("groups")) is not None:
            for group_name in groups_dtype.to_schema():
                for (group_value,), group_samples in samples.group_by(pl.col("groups").struct.field(group_name)):
                    for name, metric in self.name_to_metric.items():
                        metric_to_group_to_value[name][f"{group_name} = {group_value}"] = metric.calculate(
                            group_samples,
                            artifacts,
                        )

        return metric_to_group_to_value

    @abstractmethod
    def _prepare_samples(self, samples: pl.DataFrame, artifacts: Artifacts, /) -> pl.DataFrame: ...
