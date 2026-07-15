"""Тесты утилит: cyclopts.parse_json_mapping, logging.configure."""

import logging
import types

import pytest

from perseus.utils import logging as perseus_logging
from perseus.utils.cyclopts import parse_json_mapping


def test_parse_json_mapping_parses_single_token() -> None:
    tokens = [types.SimpleNamespace(value='{"a": 1, "b": "x"}')]
    assert parse_json_mapping(None, tokens) == {"a": 1, "b": "x"}


def test_parse_json_mapping_requires_exactly_one_token() -> None:
    with pytest.raises(ValueError, match="single string"):
        parse_json_mapping(None, [])


def test_configure_sets_up_root_logger() -> None:
    perseus_logging.configure()
    root = logging.getLogger()
    assert root.level == logging.INFO
    assert root.handlers
