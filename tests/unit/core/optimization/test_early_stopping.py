"""Тесты стейт-машины EarlyStopping."""

import pytest

from perseus.core.optimization.early_stopping import EarlyStopping


def _es(mode: str = "max", patience: int = 2, min_delta: float = 0.0) -> EarlyStopping:
    return EarlyStopping(metric="accuracy", group="overall", patience=patience, min_delta=min_delta, mode=mode)


def _metrics(value: float) -> dict[str, dict[str, float]]:
    return {"accuracy": {"overall": value}}


def test_first_epoch_never_stops_and_sets_best() -> None:
    es = _es()
    assert es.should_stop(_metrics(0.5)) is False
    assert es.best_epoch == 1
    assert es._best_value == 0.5


def test_improvement_resets_counter_max_mode() -> None:
    es = _es(mode="max", patience=2)
    es.should_stop(_metrics(0.5))
    assert es.should_stop(_metrics(0.7)) is False
    assert es.best_epoch == 2
    assert es._num_epochs_without_improvement == 0


def test_improvement_min_mode() -> None:
    es = _es(mode="min", patience=2)
    es.should_stop(_metrics(1.0))
    assert es.should_stop(_metrics(0.5)) is False
    assert es._best_value == 0.5


def test_stops_after_patience_exhausted() -> None:
    es = _es(mode="max", patience=2)
    assert es.should_stop(_metrics(0.5)) is False  # epoch 1, best
    assert es.should_stop(_metrics(0.4)) is False  # 1 без улучшения, ждём
    assert es.should_stop(_metrics(0.4)) is True  # 2 без улучшения == patience → стоп


def test_min_delta_blocks_marginal_improvement() -> None:
    es = _es(mode="max", patience=1, min_delta=0.1)
    es.should_stop(_metrics(0.5))
    # 0.55 не превышает 0.5 + 0.1 → не улучшение
    assert es.should_stop(_metrics(0.55)) is True


def test_raises_on_missing_metric() -> None:
    es = _es()
    with pytest.raises(RuntimeError, match="not found metric"):
        es.should_stop({"loss": {"overall": 0.1}})


def test_raises_on_missing_group() -> None:
    es = _es()
    with pytest.raises(RuntimeError, match="value for group"):
        es.should_stop({"accuracy": {"other_group": 0.1}})
