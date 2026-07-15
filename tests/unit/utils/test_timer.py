"""Tests for the timer decorator."""

import contextlib
import logging

from perseus.utils import timer


def test_timer_returns_value_and_logs(caplog) -> None:
    @timer(logging.getLogger("test"), "my-op")
    def add(a: int, b: int) -> int:
        return a + b

    with caplog.at_level(logging.INFO):
        result = add(2, 3)

    assert result == 5
    assert any("my-op took" in r.message for r in caplog.records)


def test_timer_logs_even_on_exception(caplog) -> None:
    @timer(logging.getLogger("test"), "boom")
    def fail() -> None:
        raise ValueError("nope")

    with caplog.at_level(logging.INFO), contextlib.suppress(ValueError):
        fail()

    assert any("boom took" in r.message for r in caplog.records)
