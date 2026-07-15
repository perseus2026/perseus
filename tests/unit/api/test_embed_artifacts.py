"""Tests for api.embed_artifacts steps: _init (dispatch) and embed_artifacts."""

import types

import polars as pl
import pytest

from perseus.api.embed_artifacts import _init, embed_artifacts
from perseus.core.tasks.classification.target import Artifacts as ClsArtifacts
from perseus.core.tasks.ranking.target import Artifacts as RankArtifacts


def _stub_checkpoint() -> types.SimpleNamespace:
    return types.SimpleNamespace(config=types.SimpleNamespace(task=types.SimpleNamespace(type="classification")))


class TestInit:
    def test_none_builds_default_artifacts(self) -> None:
        assert isinstance(_init(None, _stub_checkpoint()), ClsArtifacts)

    def test_empty_tuple(self) -> None:
        assert isinstance(_init((), _stub_checkpoint()), ClsArtifacts)

    def test_empty_dict(self) -> None:
        assert isinstance(_init({}, _stub_checkpoint()), ClsArtifacts)

    def test_passthrough_matching_instance(self) -> None:
        artifacts = ClsArtifacts()
        assert _init(artifacts, _stub_checkpoint()) is artifacts

    def test_mismatched_type_raises(self) -> None:
        with pytest.raises(TypeError, match="mismatching type"):
            _init(RankArtifacts(pl.DataFrame({"item": ["A"]})), _stub_checkpoint())


def test_embed_artifacts_returns_none_when_no_artifact_features(trained_checkpoint) -> None:
    # the minimal classification config has no artifact features → embed returns None
    assert embed_artifacts(None, trained_checkpoint) is None
