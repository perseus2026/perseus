from perseus.core.trackers.base import Tracker
from perseus.core.trackers.logging import LoggingTracker

registry: dict[str, type[Tracker]] = {
    "logging": LoggingTracker,
}

try:
    from perseus.core.trackers.clearml import ClearMLTracker
except ModuleNotFoundError:
    pass
else:
    registry["clearml"] = ClearMLTracker

__all__ = [
    "Tracker",
    "registry",
]
