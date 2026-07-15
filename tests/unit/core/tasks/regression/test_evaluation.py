"""Tests for regression metrics and Evaluator."""

import polars as pl
import pytest
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from perseus.core.tasks.regression.evaluation import R2, Evaluator, Mae, Rmse


@pytest.fixture
def samples() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "target": [1.0, 2.0, 3.0, 4.0],
            "prediction": [1.5, 2.0, 2.5, 5.0],
        },
    )


def test_rmse_matches_sklearn(samples: pl.DataFrame) -> None:
    expected = root_mean_squared_error(samples["target"], samples["prediction"])
    assert Rmse().calculate(samples, None) == pytest.approx(expected)


def test_mae_matches_sklearn(samples: pl.DataFrame) -> None:
    expected = mean_absolute_error(samples["target"], samples["prediction"])
    assert Mae().calculate(samples, None) == pytest.approx(expected)


def test_r2_matches_sklearn(samples: pl.DataFrame) -> None:
    expected = r2_score(samples["target"], samples["prediction"])
    assert R2().calculate(samples, None) == pytest.approx(expected)


def test_evaluator_predict_kwargs_empty() -> None:
    assert Evaluator({}).predict_kwargs == {}


def test_calculate_metrics_overall(samples: pl.DataFrame) -> None:
    evaluator = Evaluator({"rmse": Rmse(), "mae": Mae()})
    result = evaluator.calculate_metrics(samples, None)
    assert set(result) == {"rmse", "mae"}
    assert result["rmse"]["overall"] == pytest.approx(root_mean_squared_error(samples["target"], samples["prediction"]))
    assert result["mae"]["overall"] == pytest.approx(mean_absolute_error(samples["target"], samples["prediction"]))


def test_calculate_metrics_per_group() -> None:
    samples = pl.DataFrame(
        {
            "target": [1.0, 2.0, 3.0, 4.0],
            "prediction": [1.0, 2.0, 3.0, 4.0],
            "groups": [
                {"segment": "x"},
                {"segment": "x"},
                {"segment": "y"},
                {"segment": "y"},
            ],
        },
    )
    result = Evaluator({"mae": Mae()}).calculate_metrics(samples, None)
    assert result["mae"]["overall"] == pytest.approx(0.0)
    assert result["mae"]["segment = x"] == pytest.approx(0.0)
    assert result["mae"]["segment = y"] == pytest.approx(0.0)
