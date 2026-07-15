import typing as t
from dataclasses import dataclass
from pathlib import Path

from perseus.core import encoders, tasks
from perseus.core.backbone import Backbone
from perseus.core.config import Config


@dataclass
class Prepared:
    config: Config
    encoder_to_preprocessor: dict[str, encoders.Preprocessor]

    def save(self, path: Path, /) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.config.save(path / "config.yaml")
        for encoder, preprocessor in self.encoder_to_preprocessor.items():
            preprocessor.save(path / "encoders" / "preprocessors" / encoder)

    @classmethod
    def load(cls, path: Path, /, *, overrides: dict[str, t.Any] | None = None) -> t.Self:
        config = Config.load(path / "config.yaml", overrides=overrides)
        encoder_to_preprocessor = {
            encoder: encoders.registry[desc.type].preprocessor.load(path / "encoders" / "preprocessors" / encoder)
            for encoder, desc in config.encoders.items()
        }
        return cls(config, encoder_to_preprocessor)


@dataclass
class Trained:
    config: Config
    encoder_to_preprocessor: dict[str, encoders.Preprocessor]
    encoder_to_embedder: dict[str, encoders.Embedder]
    backbone: Backbone
    head: tasks.Head

    def save(self, path: Path, /) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.config.save(path / "config.yaml")
        for encoder, preprocessor in self.encoder_to_preprocessor.items():
            preprocessor.save(path / "encoders" / "preprocessors" / encoder)
        for encoder, embedder in self.encoder_to_embedder.items():
            embedder.save(path / "encoders" / "embedders" / encoder)
        self.backbone.save(path / "backbone")
        self.head.save(path / "head")

    @classmethod
    def load(cls, path: Path, /, *, overrides: dict[str, t.Any] | None = None) -> t.Self:
        config = Config.load(path / "config.yaml", overrides=overrides)
        encoder_to_preprocessor = {
            encoder: encoders.registry[desc.type].preprocessor.load(path / "encoders" / "preprocessors" / encoder)
            for encoder, desc in config.encoders.items()
        }
        encoder_to_embedder = {
            encoder: encoders.registry[desc.type].embedder.load(path / "encoders" / "embedders" / encoder)
            for encoder, desc in config.encoders.items()
        }
        backbone = Backbone.load(path / "backbone")
        head = tasks.registry[config.task.type].head.load(path / "head")
        return cls(config, encoder_to_preprocessor, encoder_to_embedder, backbone, head)
