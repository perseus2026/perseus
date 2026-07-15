import typing as t
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import torch
from sklearn.base import BaseEstimator
from sklearn.preprocessing import (
    FunctionTransformer,
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from torch import nn

from perseus.core import encoders
from perseus.core.tasks import base


class Artifacts(base.Artifacts):
    def static_transform(self, _feature_to_preprocessor: dict[str, encoders.Preprocessor], /) -> t.Self:
        return self

    def dynamic_transform(self, _feature_to_preprocessor: dict[str, encoders.Preprocessor], /) -> None:
        return

    def embed(
        self,
        _transformed: t.Any,
        _feature_to_embedder: dict[str, encoders.Embedder],
        _target: t.Any | None = None,
        /,
    ) -> None:
        return

    def save(self, _path: Path, /) -> None:
        return

    @classmethod
    def load(cls, _path: Path, /) -> t.Self:
        return cls()


class Preprocessor(base.Preprocessor):
    def __init__(self, scaler: BaseEstimator, /) -> None:
        self.scaler = scaler

    @classmethod
    def fit(
        cls,
        values: pl.Series,
        _artifacts: Artifacts,
        _feature_to_observer: dict[str, encoders.Observer],
        /,
        *,
        scaler: t.Literal["none", "log", "standard", "min-max", "max-abs", "robust", "quantile", "power"] = "none",
        scaler_params: dict[str, t.Any] | None = None,
    ) -> t.Self:
        match scaler:
            case "none":
                scaler_cls = FunctionTransformer
            case "log":
                scaler_cls = partial(FunctionTransformer, func=np.log1p, inverse_func=np.expm1)
            case "standard":
                scaler_cls = StandardScaler
            case "min-max":
                scaler_cls = MinMaxScaler
            case "max-abs":
                scaler_cls = MaxAbsScaler
            case "robust":
                scaler_cls = RobustScaler
            case "quantile":
                scaler_cls = QuantileTransformer
            case "power":
                scaler_cls = PowerTransformer
            case _:
                raise ValueError(f"unknown {scaler = }")

        return cls(scaler_cls(**(scaler_params or {})).fit(values.to_numpy().reshape(-1, 1)))

    def static_transform(self, values: pl.Series, /) -> pl.Series:
        return pl.Series(
            name="value",
            values=self.scaler.transform(values.to_numpy().reshape(-1, 1)).squeeze(),
            dtype=pl.Float32(),
        )

    def dynamic_transform(self, value: float, /) -> float:
        return value

    def collate(self, values: list[float], /) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32)

    def save(self, path: Path, /) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path / "scaler.jbl")

    @classmethod
    def load(cls, path: Path, /) -> t.Self:
        return cls(joblib.load(path / "scaler.jbl"))


class Layer(base.Layer):
    def __init__(self, dim: int, /) -> None:
        super().__init__()

        self.linear = nn.Linear(dim, 1)

    @classmethod
    def init(cls, _preprocessor: Preprocessor, dim: int, /) -> t.Self:
        return cls(dim)

    def forward(self, backbone_embeddings: torch.Tensor, _embedded_artifacts: t.Any, /) -> torch.Tensor:
        return self.linear(backbone_embeddings)


class Criterion(base.Criterion):
    def __init__(
        self,
        layer: Layer,
        /,
        *,
        loss_fn: t.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()

        self._layer = [layer]

        self.loss_fn = loss_fn

    @classmethod
    def init(
        cls,
        _preprocessor: Preprocessor,
        layer: Layer,
        /,
        *,
        loss: t.Literal["mse", "l1", "smooth_l1", "huber"] = "mse",
        loss_params: dict[str, t.Any] | None = None,
    ) -> t.Self:
        match loss:
            case "mse":
                loss_fn = nn.functional.mse_loss
            case "l1":
                loss_fn = nn.functional.l1_loss
            case "smooth_l1":
                loss_fn = nn.functional.smooth_l1_loss
            case "huber":
                loss_fn = nn.functional.huber_loss
            case _:
                raise ValueError(f"unknown loss = {loss}")

        loss_fn = partial(loss_fn, **(loss_params or {}))
        return cls(layer, loss_fn=loss_fn)

    def forward(
        self,
        backbone_embeddings: torch.Tensor,
        embedded_artifacts: t.Any,
        target: torch.Tensor,
        /,
    ) -> torch.Tensor:
        prediction = self._layer[0](backbone_embeddings, embedded_artifacts)
        return self.loss_fn(prediction.squeeze(dim=1), target)


class Head(base.Head):
    def __init__(self, layer: Layer, scaler: BaseEstimator, /) -> None:
        self.layer = layer
        self.scaler = scaler

    @classmethod
    def init(cls, preprocessor: Preprocessor, layer: Layer, /) -> t.Self:
        return cls(layer, preprocessor.scaler)

    def predict(
        self,
        backbone_embeddings: torch.Tensor,
        embedded_artifacts: t.Any,
        _extras: dict[str, t.Any],
        /,
    ) -> pl.Series:
        self.layer.eval().to(backbone_embeddings.device)

        prediction = self.layer(backbone_embeddings, embedded_artifacts).to(torch.float32).cpu().numpy()
        prediction = self.scaler.inverse_transform(prediction)[:, 0]
        return pl.Series("prediction", prediction, pl.Float32())

    def save(self, path: Path, /) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.layer.save(path / "layer")
        joblib.dump(self.scaler, path / "scaler.jbl")

    @classmethod
    def load(cls, path: Path, /) -> t.Self:
        layer = Layer.load(path / "layer")
        scaler = joblib.load(path / "scaler.jbl")
        return cls(layer, scaler)
