from perseus.core.backbone.event import concat, weighted_sum
from perseus.core.backbone.event import sum as sum_
from perseus.core.backbone.event.base import Aggregator

__all__ = [
    "Aggregator",
    "registry",
]

registry: dict[str, type[Aggregator]] = {
    "sum": sum_.Aggregator,
    "weighted_sum": weighted_sum.Aggregator,
    "concat": concat.Aggregator,
}
