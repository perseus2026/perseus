from perseus.core.backbone.context import identity
from perseus.core.backbone.context.base import Aggregator

__all__ = [
    "Aggregator",
    "registry",
]

registry: dict[str, type[Aggregator]] = {
    "identity": identity.Aggregator,
}
