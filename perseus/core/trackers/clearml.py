import clearml

from perseus.core.trackers.base import Tracker


class ClearMLTracker(Tracker):
    def __init__(
        self,
        *,
        project: str | None = None,
        name: str | None = None,
        reuse: bool = False,
        tags: list[str] | None = None,
    ) -> None:
        self._task = clearml.Task.init(
            project_name=project,
            task_name=name,
            reuse_last_task_id=reuse,
            tags=tags,
            auto_connect_arg_parser=False,
            auto_connect_frameworks=False,
        )

    def log_metrics(self, metrics: dict[str, dict[str, float]], /, *, iteration: int) -> None:
        logger = self._task.get_logger()
        for metric_name, group_metrics in metrics.items():
            for group_name, metric_value in group_metrics.items():
                logger.report_scalar(title=metric_name, series=group_name, value=metric_value, iteration=iteration)
