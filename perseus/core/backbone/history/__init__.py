from perseus.core.backbone.history import bert, danet, hstu, ligr, mamba, modern_bert
from perseus.core.backbone.history.base import Aggregator

__all__ = [
    "Aggregator",
    "registry",
]

registry: dict[str, type[Aggregator]] = {
    "bert": bert.Aggregator,
    "modern_bert": modern_bert.Aggregator,
    "hstu": hstu.Aggregator,
    "mamba": mamba.Aggregator,
    "danet": danet.Aggregator,
    "ligr": ligr.Aggregator,
}
