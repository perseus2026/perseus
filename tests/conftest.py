"""Shared fixtures for all perseus tests."""

import copy
import typing as t
from pathlib import Path

import pytest
import torch
import yaml

# Minimal valid config: one classification task, one event, backbone.
# model_post_init adds the "event" encoder/feature itself, so they are not needed here.
MINIMAL_CONFIG: dict[str, t.Any] = {
    "task": {
        "type": "classification",
        "metrics": {"accuracy": None},
    },
    "events": {"purchase": None},
    "backbone": {"dim": 8},
}


@pytest.fixture
def minimal_config_dict() -> dict[str, t.Any]:
    """A fresh copy of the minimal valid config (dict)."""
    return copy.deepcopy(MINIMAL_CONFIG)


@pytest.fixture
def minimal_config_path(tmp_path: Path, minimal_config_dict: dict[str, t.Any]) -> Path:
    """The minimal config written as YAML to a temporary file."""
    path = tmp_path / "config.yaml"
    with path.open("w") as f:
        yaml.safe_dump(minimal_config_dict, f, allow_unicode=True, sort_keys=False)
    return path


@pytest.fixture(autouse=True)
def _seed_torch() -> None:
    """Deterministic torch for all tests."""
    torch.manual_seed(0)
