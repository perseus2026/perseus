"""Step-by-step tests for api.prepare_dataset."""

import datetime as dt

import polars as pl
import pytest

from perseus.api.prepare_dataset import (
    _distribute_samples,
    _filter_samples,
    _fit_encoders_preprocessors,
    _fit_task_preprocessor,
    _init_artifacts,
    _observe_events,
    _observe_samples,
    _static_transform_basis,
    _static_transform_events,
)
from perseus.core.encoders.id import Observer as IdObserver
from perseus.core.tasks.classification.target import Artifacts as ClsArtifacts
from perseus.core.tasks.ranking.target import Artifacts as RankArtifacts
from tests.unit.api.conftest import make_config


def _observers() -> dict:
    return {"event": IdObserver()}


class TestInitArtifacts:
    def test_none(self) -> None:
        assert isinstance(_init_artifacts(None, make_config()), ClsArtifacts)

    def test_tuple_and_dict(self) -> None:
        assert isinstance(_init_artifacts((), make_config()), ClsArtifacts)
        assert isinstance(_init_artifacts({}, make_config()), ClsArtifacts)

    def test_mismatched_raises(self) -> None:
        with pytest.raises(TypeError, match="mismatching type"):
            _init_artifacts(RankArtifacts(pl.DataFrame({"item": ["A"]})), make_config())


def test_distribute_samples_concats_folds() -> None:
    train = pl.DataFrame({"client_id": ["c1"], "timestamp": [dt.datetime(2024, 1, 1)]})
    test = pl.DataFrame({"client_id": ["c2"], "timestamp": [dt.datetime(2024, 1, 1)]})
    out = list(_distribute_samples(train, test))
    folds = {fold for _, part in out for fold in part["_fold"].to_list()}
    assert folds == {"train", "test"}
    for _, part in out:
        assert "partition_id" not in part.schema
        assert "_sample_timestamp" in part.schema


def test_observe_events_observes_train_only() -> None:
    events = pl.DataFrame(
        {
            "client_id": ["c1", "c1"],
            "timestamp": [dt.datetime(2024, 1, 1), dt.datetime(2024, 3, 1)],
            "event": ["purchase", "view"],
        },
    )
    samples = pl.DataFrame(
        {
            "client_id": ["c1"],
            "_sample_timestamp": [dt.datetime(2024, 2, 1)],
            "_fold": ["train"],
        },
    )
    observers = _observers()
    _observe_events(events, samples, observers, make_config())
    # only the event before sample_timestamp is observed (purchase, 2024-01)
    assert observers["event"].value_counts["value"].to_list() == ["purchase"]


def test_filter_samples_keeps_only_after_first_event() -> None:
    first_ts = pl.DataFrame({"client_id": ["c1"], "_first_event_timestamp": [dt.datetime(2024, 1, 1)]})
    train = pl.DataFrame({"client_id": ["c1"], "timestamp": [dt.datetime(2024, 2, 1)]})
    test = pl.DataFrame({"client_id": ["c1"], "timestamp": [dt.datetime(2023, 1, 1)]})
    train_out, test_out = _filter_samples(first_ts, train, test)
    assert train_out.height == 1
    assert test_out.height == 0  # sample earlier than the first event


def test_observe_samples_noop_without_context() -> None:
    observers = _observers()
    _observe_samples(pl.DataFrame({"client_id": ["c1"]}), observers, make_config())
    assert observers["event"].value_counts.is_empty()


def test_fit_task_preprocessor() -> None:
    train = pl.DataFrame({"target": ["a", "b", "a"]})
    pre = _fit_task_preprocessor(train, ClsArtifacts(), _observers(), make_config())
    assert pre.label_to_index == {"a": 0, "b": 1}


def test_fit_encoders_preprocessors() -> None:
    observers = _observers()
    observers["event"].observe(pl.Series(["purchase", "view", "purchase"]))
    preprocessors = _fit_encoders_preprocessors(observers, make_config())
    assert "event" in preprocessors
    assert set(preprocessors["event"].values["value"]) == {"purchase", "view"}


def test_static_transform_events() -> None:
    observers = _observers()
    observers["event"].observe(pl.Series(["purchase", "view"]))
    preprocessors = _fit_encoders_preprocessors(observers, make_config())
    events = pl.DataFrame({"event": ["purchase", "view"]})
    out = _static_transform_events(events, preprocessors, make_config())
    # event names are replaced with integer indices
    assert out["event"].dtype == pl.UInt32


def test_static_transform_basis_transforms_target_and_sorts() -> None:
    observers = _observers()
    observers["event"].observe(pl.Series(["purchase"]))
    preprocessors = _fit_encoders_preprocessors(observers, make_config())
    task_pre = _fit_task_preprocessor(pl.DataFrame({"target": ["a", "b"]}), ClsArtifacts(), observers, make_config())

    train = pl.DataFrame(
        {
            "target": ["b", "a"],
            "client_id": ["c2", "c1"],
            "timestamp": [dt.datetime(2024, 1, 2), dt.datetime(2024, 1, 1)],
        },
    )
    test = pl.DataFrame({"client_id": ["c3"], "timestamp": [dt.datetime(2024, 1, 1)]})
    train_out, _, _test_out, _ = _static_transform_basis(
        train,
        ClsArtifacts(),
        test,
        ClsArtifacts(),
        preprocessors,
        task_pre,
        make_config(),
    )
    # target is converted to indices and sorted by timestamp (c1 before c2)
    assert train_out["client_id"].to_list() == ["c1", "c2"]
    assert train_out["target"].to_list() == [0, 1]
