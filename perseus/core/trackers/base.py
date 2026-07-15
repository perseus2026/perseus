from abc import ABC, abstractmethod


class Tracker(ABC):
    @abstractmethod
    def log_metrics(self, metrics: dict[str, dict[str, float]], /, *, iteration: int) -> None: ...
