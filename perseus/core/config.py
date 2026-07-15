import logging
import typing as t
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Metric(BaseModel):
    type: str | None = None
    params: t.Annotated[dict[str, t.Any], Field(default_factory=dict)]


class Task(BaseModel):
    type: str
    metrics: dict[str, Metric | None]
    preprocessor: t.Annotated[dict[str, t.Any], Field(default_factory=dict)]
    layer: t.Annotated[dict[str, t.Any], Field(default_factory=dict)]
    criterion: t.Annotated[dict[str, t.Any], Field(default_factory=dict)]
    head: t.Annotated[dict[str, t.Any], Field(default_factory=dict)]

    def model_post_init(self, _context: t.Any) -> None:
        self.metrics = {name: (Metric() if metric is None else metric) for name, metric in self.metrics.items()}
        for name, metric in self.metrics.items():
            if metric.type is None:
                metric.type = name


class Attribute(BaseModel):
    multi: bool = False
    field: str | None = None


class Event(BaseModel):
    attributes: t.Annotated[dict[str, Attribute | None], Field(default_factory=dict)]
    max_tokens: int | None = None
    priority: int = 0
    max_events_per_sequence: int | None = None
    max_duration_per_sequence: str | None = None
    source: str = "event_hub"

    def model_post_init(self, _context: t.Any) -> None:
        self.attributes = {name: (Attribute() if attr is None else attr) for name, attr in self.attributes.items()}
        for name, attr in self.attributes.items():
            if attr.field is None:
                attr.field = name
        self.attributes["event"] = Attribute()

        if any(attr.multi for attr in self.attributes.values()) and self.max_tokens is None:
            raise ValueError("max_tokens must be set if at least one attribute is multi")


class Encoder(BaseModel):
    type: str
    preprocessor: t.Annotated[dict[str, t.Any], Field(default_factory=dict)]
    embedder: t.Annotated[dict[str, t.Any], Field(default_factory=dict)]


class LocatedIn(BaseModel):
    event: bool = False
    context: bool = False
    artifacts: bool = False

    def model_post_init(self, _context: t.Any) -> None:
        if not any([self.event, self.context, self.artifacts]):
            raise ValueError("at least 1 location of event, context or artifacts must be specified")


class Feature(BaseModel):
    located_in: LocatedIn
    encoder: str | Encoder


class Aggregator(BaseModel):
    type: str
    params: t.Annotated[dict[str, t.Any], Field(default_factory=dict)]


class Backbone(BaseModel):
    dim: int
    history_aggregator: t.Annotated[Aggregator, Field(default_factory=lambda: Aggregator(type="modern_bert"))]
    event_aggregator: t.Annotated[Aggregator, Field(default_factory=lambda: Aggregator(type="weighted_sum"))]
    context_aggregator: Aggregator | None = None


class Tracker(BaseModel):
    type: str = "logging"
    params: t.Annotated[dict[str, t.Any], Field(default_factory=dict)]


class DataLoader(BaseModel):
    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = False


class TrainDataLoader(DataLoader):
    shuffle: bool = False


class Optimizer(BaseModel):
    type: str = "adamw"
    params: t.Annotated[dict[str, t.Any], Field(default_factory=dict)]


class Scheduler(BaseModel):
    type: str = "constant"
    params: t.Annotated[dict[str, t.Any], Field(default_factory=dict)]


class EarlyStopping(BaseModel):
    metric: str
    group: str = "overall"
    mode: t.Literal["min", "max"] = "max"
    patience: int = 1
    min_delta: float = 0.0


class Training(BaseModel):
    num_epochs: int = 1
    log_every_n_train_steps: int = 10_000

    tracker: t.Annotated[Tracker, Field(default_factory=Tracker)]
    dataloader: t.Annotated[TrainDataLoader, Field(default_factory=TrainDataLoader)]
    test_dataloader: DataLoader | None = None
    optimizer: t.Annotated[Optimizer, Field(default_factory=Optimizer)]
    scheduler: t.Annotated[Scheduler, Field(default_factory=Scheduler)]
    early_stopping: EarlyStopping | None = None

    def model_post_init(self, _context: t.Any) -> None:
        if self.test_dataloader is None:
            self.test_dataloader = DataLoader(
                batch_size=self.dataloader.batch_size,
                num_workers=self.dataloader.num_workers,
                pin_memory=self.dataloader.pin_memory,
            )


class Inference(BaseModel):
    event_store_backend: t.Literal["disk", "ram"] = "ram"
    predict_kwargs: t.Annotated[dict[str, t.Any], Field(default_factory=dict)]
    backbone_dataloader: t.Annotated[DataLoader, Field(default_factory=DataLoader)]
    head_dataloader: t.Annotated[DataLoader, Field(default_factory=DataLoader)]


class Config(BaseModel):
    task: Task
    events: dict[str, Event | None]
    max_events_per_sequence: int = 512
    max_duration_per_sequence: str | None = None
    encoders: t.Annotated[dict[str, Encoder], Field(default_factory=dict)]
    features: t.Annotated[dict[str, Feature], Field(default_factory=dict)]
    backbone: Backbone
    training: t.Annotated[Training, Field(default_factory=Training)]
    inference: t.Annotated[Inference, Field(default_factory=Inference)]

    def model_post_init(self, _context: t.Any) -> None:
        self.events = {event: (Event() if desc is None else desc) for event, desc in self.events.items()}

        self.encoders["event"] = Encoder(type="id")
        self.features["event"] = Feature(located_in=LocatedIn(event=True), encoder="event")
        for feature, desc in self.features.items():
            if isinstance(desc.encoder, Encoder):
                if feature in self.encoders:
                    raise ValueError(f"encoder `{feature}` already exists")
                self.encoders[feature] = desc.encoder
                desc.encoder = feature

        declared_attributes = {name for event in self.events.values() for name in event.attributes}
        feature_attributes = set(self.features.keys())
        if unused_attributes := declared_attributes - feature_attributes:
            raise ValueError(f"event attributes are declared, but unused: {', '.join(unused_attributes)}")

        declared_encoders = set(self.encoders.keys())
        feature_encoders = {feature.encoder for feature in self.features.values()}
        if unused_encoders := declared_encoders - feature_encoders:
            raise ValueError(f"encoders are declared, but unused: {', '.join(unused_encoders)}")
        if unknown_encoders := feature_encoders - declared_encoders:
            raise ValueError(f"unknown encoders: {', '.join(unknown_encoders)}")

        num_context_features = len(self.feature_to_encoder(context=True))
        if num_context_features > 0 and self.backbone.context_aggregator is None:
            raise ValueError("declared context features, but context aggregator is not defined")
        if num_context_features == 0 and self.backbone.context_aggregator is not None:
            raise ValueError("declared no context features, but context aggregator is defined")

    def feature_to_encoder(
        self,
        *,
        event: bool = False,
        context: bool = False,
        artifacts: bool = False,
    ) -> dict[str, str]:
        if not any([event, context, artifacts]):
            raise ValueError("must set at least one location from `event`, `context` and `artifacts`")
        return {
            feature: desc.encoder
            for feature, desc in self.features.items()
            if (desc.located_in.event and event)
            or (desc.located_in.context and context)
            or (desc.located_in.artifacts and artifacts)
        }

    def save(self, path: Path, /) -> None:
        with path.open("w") as f:
            yaml.safe_dump(self.model_dump(), f, indent=2, sort_keys=False, allow_unicode=True)

    @classmethod
    def load(cls, path: Path, /, *, overrides: dict[str, t.Any] | None = None) -> t.Self:
        with path.open() as f:
            base = yaml.safe_load(f)
        if overrides:
            base = _deep_merge(base, overrides)
        config = cls.model_validate(base)
        logger.info(
            "loaded config:\n%s",
            yaml.safe_dump(config.model_dump(), indent=2, sort_keys=False, allow_unicode=True),
        )
        return config


def _deep_merge(base: dict[str, t.Any], override: dict[str, t.Any], /) -> dict[str, t.Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
