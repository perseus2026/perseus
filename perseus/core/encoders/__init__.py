from perseus.core.encoders import bow, ple
from perseus.core.encoders import id as id_
from perseus.core.encoders.base import Embedder, Encoder, Observer, Preprocessor

__all__ = [
    "Embedder",
    "Encoder",
    "Observer",
    "Preprocessor",
    "registry",
]


registry: dict[str, Encoder] = {
    "id": id_.encoder,
    "ple": ple.encoder,
    "bow": bow.encoder,
}
