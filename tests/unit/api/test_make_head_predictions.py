"""Пошаговые тесты api.make_head_predictions."""

import polars as pl

from perseus.api.make_head_predictions import _create_dataloader, _join_predictions, _make_predictions
from tests.unit.api.conftest import DIM


def _embeddings() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "_index": [0, 1, 2],
            "embeddings": [[0.1] * DIM, [0.2] * DIM, [0.3] * DIM],
        },
        schema_overrides={"embeddings": pl.Array(pl.Float32, DIM)},
    )


def _samples() -> pl.DataFrame:
    return pl.DataFrame({"_index": [0, 1, 2], "client_id": ["c1", "c2", "c3"]})


def test_create_dataloader_and_make_predictions(trained_checkpoint) -> None:
    dataloader = _create_dataloader(_samples(), _embeddings(), trained_checkpoint)
    predictions = _make_predictions(dataloader, None, trained_checkpoint)

    assert set(predictions.columns) == {"_index", "prediction"}
    assert predictions.height == 3
    # classification head → struct вероятностей по меткам a, b
    assert predictions["prediction"].dtype == pl.Struct({"a": pl.Float32, "b": pl.Float32})


def test_join_predictions() -> None:
    samples = _samples()
    predictions = pl.DataFrame({"_index": [0, 1, 2], "predictions": [0.5, 0.6, 0.7]})
    out = _join_predictions(samples, predictions)
    assert out.height == 3
    assert "predictions" in out.columns
    assert "client_id" in out.columns
