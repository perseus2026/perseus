import logging
import typing as t

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(
        self,
        metric: str,
        group: str,
        patience: int,
        min_delta: float,
        mode: t.Literal["min", "max"],
    ) -> None:
        self.metric = metric
        self.group = group

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_epoch = 0
        self._best_value: float
        self._num_epochs_without_improvement = 0

    def should_stop(self, metrics: dict[str, dict[str, float]], /) -> bool:
        if self._num_epochs_without_improvement > self.patience:
            raise RuntimeError("must stop earlier")

        if (groups := metrics.get(self.metric)) is None:
            raise RuntimeError(f"not found metric {self.metric}")
        if (value := groups.get(self.group)) is None:
            raise RuntimeError(f"not found {self.metric} value for group {self.group}")

        epoch = self.best_epoch + self._num_epochs_without_improvement + 1
        logger.info("epoch = %s: %s %s = %s", epoch, self.group, self.metric, value)

        if epoch == 1:
            self._best_value = value
            self.best_epoch = 1
            return False

        improved = (
            value < self._best_value - self.min_delta
            if self.mode == "min"
            else value > self._best_value + self.min_delta
        )
        if improved:
            logger.info("%s %s improved from previous best value = %s", self.group, self.metric, self._best_value)
            self._best_value = value
            self._num_epochs_without_improvement = 0
            self.best_epoch = epoch
            return False

        self._num_epochs_without_improvement += 1
        if (patience_left := self.patience - self._num_epochs_without_improvement) > 0:
            logger.info(
                "%s %s does not improve for %s epochs, waiting for %s more epochs to improve from best value = %s",
                self.group,
                self.metric,
                self._num_epochs_without_improvement,
                patience_left,
                self._best_value,
            )
            return False

        logger.info(
            "%s %s does not improve for %s epochs, stop training with best value = %s (best epoch = %s)",
            self.group,
            self.metric,
            self.patience,
            self._best_value,
            self.best_epoch,
        )
        return True
