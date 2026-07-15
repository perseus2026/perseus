import logging

from perseus.core.trackers.base import Tracker

logger = logging.getLogger(__name__)


class LoggingTracker(Tracker):
    def log_metrics(self, metrics: dict[str, dict[str, float]], /, *, iteration: int) -> None:
        for metric_name, group_metrics in metrics.items():
            for group_name, metric_value in group_metrics.items():
                logger.info("[iteration=%d] %s/%s = %s", iteration, metric_name, group_name, metric_value)
