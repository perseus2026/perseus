"""Tests for classification metrics and Evaluator."""

import polars as pl
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from perseus.core.tasks.classification.evaluation import (
    F1,
    Accuracy,
    Evaluator,
    PrAuc,
    Precision,
    Recall,
    RocAuc,
)


@pytest.fixture
def binary_samples() -> pl.DataFrame:
    # target in {a, b}; prediction is a struct with per-label probabilities
    return pl.DataFrame(
        {
            "target": ["a", "b", "a", "b"],
            "prediction": [
                {"a": 0.9, "b": 0.1},  # -> a (correct)
                {"a": 0.2, "b": 0.8},  # -> b (correct)
                {"a": 0.4, "b": 0.6},  # -> b (incorrect)
                {"a": 0.3, "b": 0.7},  # -> b (correct)
            ],
        },
    )


@pytest.fixture
def prepared(binary_samples: pl.DataFrame) -> pl.DataFrame:
    # most_probable_label is computed in _prepare_samples
    return Evaluator({})._prepare_samples(binary_samples, None)


def test_prepare_samples_adds_most_probable_label(prepared: pl.DataFrame) -> None:
    assert prepared["most_probable_label"].to_list() == ["a", "b", "b", "b"]


def test_accuracy_matches_sklearn(prepared: pl.DataFrame) -> None:
    expected = accuracy_score(prepared["target"], prepared["most_probable_label"])
    assert Accuracy().calculate(prepared, None) == pytest.approx(expected)
    assert Accuracy().calculate(prepared, None) == pytest.approx(0.75)


def test_recall_precision_f1_match_sklearn(prepared: pl.DataFrame) -> None:
    y_true, y_pred = prepared["target"], prepared["most_probable_label"]
    assert Recall(pos_label="a").calculate(prepared, None) == pytest.approx(
        recall_score(y_true, y_pred, pos_label="a"),
    )
    assert Precision(pos_label="a").calculate(prepared, None) == pytest.approx(
        precision_score(y_true, y_pred, pos_label="a"),
    )
    assert F1(pos_label="a").calculate(prepared, None) == pytest.approx(
        f1_score(y_true, y_pred, pos_label="a"),
    )


def test_threshold_path_binarizes_target(prepared: pl.DataFrame) -> None:
    # threshold: y_true = (target == pos_label), y_pred = (P(pos) > threshold) -- both bool.
    # P(b) = [0.1, 0.8, 0.6, 0.7] > 0.5 -> [F, T, T, T]; target == b -> [F, T, F, T] -> accuracy 3/4
    metric = Accuracy(threshold=0.5, pos_label="b")
    expected = accuracy_score([False, True, False, True], [False, True, True, True])
    assert metric.calculate(prepared, None) == pytest.approx(expected)
    assert metric.calculate(prepared, None) == pytest.approx(0.75)


def test_threshold_path_with_pos_label_metric(prepared: pl.DataFrame) -> None:
    # for recall_score (which has a pos_label parameter) it is replaced with True after binarization
    metric = Recall(threshold=0.5, pos_label="b")
    # y_true=[F,T,F,T], y_pred=[F,T,T,T] -> recall for class True: TP=2, FN=0 -> 1.0
    assert metric.calculate(prepared, None) == pytest.approx(1.0)


def test_threshold_requires_binary() -> None:
    samples = pl.DataFrame(
        {
            "target": ["a", "b", "c"],
            "most_probable_label": ["a", "b", "c"],
            "prediction": [
                {"a": 1.0, "b": 0.0, "c": 0.0},
                {"a": 0.0, "b": 1.0, "c": 0.0},
                {"a": 0.0, "b": 0.0, "c": 1.0},
            ],
        },
    )
    with pytest.raises(ValueError, match="binary classification"):
        Accuracy(threshold=0.5, pos_label="a").calculate(samples, None)


def test_threshold_requires_pos_label(prepared: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="pos_label"):
        Accuracy(threshold=0.5).calculate(prepared, None)


def test_roc_auc_binary(binary_samples: pl.DataFrame) -> None:
    value = RocAuc(pos_label="b").calculate(binary_samples, None)
    assert 0.0 <= value <= 1.0


def test_roc_auc_binary_requires_pos_label(binary_samples: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="label must be set"):
        RocAuc().calculate(binary_samples, None)


def test_pr_auc_binary(binary_samples: pl.DataFrame) -> None:
    value = PrAuc(pos_label="b").calculate(binary_samples, None)
    assert 0.0 <= value <= 1.0


def test_pr_auc_binary_requires_pos_label(binary_samples: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="label must be set"):
        PrAuc().calculate(binary_samples, None)


def test_multiclass_roc_auc_and_pr_auc() -> None:
    samples = pl.DataFrame(
        {
            "target": ["a", "b", "c", "a"],
            "prediction": [
                {"a": 0.7, "b": 0.2, "c": 0.1},
                {"a": 0.1, "b": 0.8, "c": 0.1},
                {"a": 0.1, "b": 0.2, "c": 0.7},
                {"a": 0.6, "b": 0.3, "c": 0.1},
            ],
        },
    )
    roc = RocAuc(multi_class="ovr").calculate(samples, None)
    pr = PrAuc().calculate(samples, None)
    assert 0.0 <= roc <= 1.0
    assert 0.0 <= pr <= 1.0


def test_calculate_metrics_overall(binary_samples: pl.DataFrame) -> None:
    evaluator = Evaluator({"accuracy": Accuracy(), "roc_auc": RocAuc(pos_label="b")})
    result = evaluator.calculate_metrics(binary_samples, None)
    assert result["accuracy"]["overall"] == pytest.approx(0.75)
    assert 0.0 <= result["roc_auc"]["overall"] <= 1.0
