"""Общие фикстуры для всех тестов perseus."""

import copy
import typing as t
from pathlib import Path

import pytest
import torch
import yaml

# Минимальный валидный конфиг: одна задача classification, одно событие, backbone.
# model_post_init сам добавит encoder/feature "event", поэтому здесь они не нужны.
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
    """Свежая копия минимального валидного конфига (dict)."""
    return copy.deepcopy(MINIMAL_CONFIG)


@pytest.fixture
def minimal_config_path(tmp_path: Path, minimal_config_dict: dict[str, t.Any]) -> Path:
    """Минимальный конфиг, записанный в YAML во временный файл."""
    path = tmp_path / "config.yaml"
    with path.open("w") as f:
        yaml.safe_dump(minimal_config_dict, f, allow_unicode=True, sort_keys=False)
    return path


@pytest.fixture(autouse=True)
def _seed_torch() -> None:
    """Детерминированность torch для всех тестов."""
    torch.manual_seed(0)
