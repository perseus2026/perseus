import datetime as dt
import time
import typing as t
from functools import wraps
from logging import Logger

from perseus.utils import cyclopts, logging, torch

__all__ = [
    "cyclopts",
    "logging",
    "torch",
]

P = t.ParamSpec("P")
R = t.TypeVar("R")


def timer(logger: Logger, name: str, /) -> t.Callable[[t.Callable[P, R]], t.Callable[P, R]]:
    def decorator(func: t.Callable[P, R]) -> t.Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            started_at = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                ended_at = time.perf_counter()
                time_spent = dt.timedelta(seconds=ended_at - started_at)
                logger.info("%s took %s", name, time_spent)

        return wrapper

    return decorator
