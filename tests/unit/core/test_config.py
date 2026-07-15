"""Тесты pydantic-конфига: валидация, инварианты model_post_init, save/load, merge."""

import typing as t

import pytest

from perseus.core.config import Config, _deep_merge


def test_minimal_config_validates(minimal_config_dict: dict[str, t.Any]) -> None:
    config = Config.model_validate(minimal_config_dict)
    # event-encoder и event-feature добавляются автоматически
    assert "event" in config.encoders
    assert config.encoders["event"].type == "id"
    assert "event" in config.features
    assert config.features["event"].located_in.event is True


def test_metric_type_defaults_to_name(minimal_config_dict: dict[str, t.Any]) -> None:
    config = Config.model_validate(minimal_config_dict)
    assert config.task.metrics["accuracy"].type == "accuracy"


def test_inline_encoder_is_lifted_into_encoders(minimal_config_dict: dict[str, t.Any]) -> None:
    minimal_config_dict["features"] = {
        "city": {
            "located_in": {"event": True},
            "encoder": {"type": "id", "preprocessor": {}, "embedder": {}},
        },
    }
    minimal_config_dict["events"] = {"purchase": {"attributes": {"city": None}}}
    config = Config.model_validate(minimal_config_dict)
    # inline-энкодер переехал в encoders под именем фичи, а ссылка стала строкой
    assert "city" in config.encoders
    assert config.features["city"].encoder == "city"


def test_multi_attribute_requires_max_tokens(minimal_config_dict: dict[str, t.Any]) -> None:
    minimal_config_dict["events"] = {"purchase": {"attributes": {"items": {"multi": True}}}}
    with pytest.raises(ValueError, match="max_tokens"):
        Config.model_validate(minimal_config_dict)


def test_unused_attribute_raises(minimal_config_dict: dict[str, t.Any]) -> None:
    # объявлен атрибут "city", но нет фичи для него
    minimal_config_dict["events"] = {"purchase": {"attributes": {"city": None}}}
    with pytest.raises(ValueError, match="unused"):
        Config.model_validate(minimal_config_dict)


def test_unknown_encoder_raises(minimal_config_dict: dict[str, t.Any]) -> None:
    minimal_config_dict["features"] = {
        "city": {"located_in": {"event": True}, "encoder": "no_such_encoder"},
    }
    minimal_config_dict["events"] = {"purchase": {"attributes": {"city": None}}}
    with pytest.raises(ValueError, match="unknown encoders"):
        Config.model_validate(minimal_config_dict)


def test_context_feature_requires_aggregator(minimal_config_dict: dict[str, t.Any]) -> None:
    minimal_config_dict["encoders"] = {"city": {"type": "id"}}
    minimal_config_dict["features"] = {
        "city": {"located_in": {"context": True}, "encoder": "city"},
    }
    with pytest.raises(ValueError, match="context aggregator is not defined"):
        Config.model_validate(minimal_config_dict)


def test_located_in_requires_at_least_one_location(minimal_config_dict: dict[str, t.Any]) -> None:
    minimal_config_dict["encoders"] = {"city": {"type": "id"}}
    minimal_config_dict["features"] = {
        "city": {"located_in": {}, "encoder": "city"},
    }
    minimal_config_dict["events"] = {"purchase": {"attributes": {"city": None}}}
    with pytest.raises(ValueError, match="at least 1 location"):
        Config.model_validate(minimal_config_dict)


def test_feature_to_encoder_filters_by_location(minimal_config_dict: dict[str, t.Any]) -> None:
    config = Config.model_validate(minimal_config_dict)
    event_encoders = config.feature_to_encoder(event=True)
    assert event_encoders == {"event": "event"}

    with pytest.raises(ValueError, match="at least one location"):
        config.feature_to_encoder()


def test_save_load_roundtrip(minimal_config_dict, tmp_path) -> None:
    config = Config.model_validate(minimal_config_dict)
    path = tmp_path / "out.yaml"
    config.save(path)
    loaded = Config.load(path)
    assert loaded.model_dump() == config.model_dump()


def test_load_applies_overrides(minimal_config_path) -> None:
    loaded = Config.load(minimal_config_path, overrides={"backbone": {"dim": 16}})
    assert loaded.backbone.dim == 16


class TestDeepMerge:
    def test_overrides_scalar(self) -> None:
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_merges_nested_dicts(self) -> None:
        assert _deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}

    def test_adds_new_keys(self) -> None:
        assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_does_not_mutate_base(self) -> None:
        base = {"a": {"b": 1}}
        _deep_merge(base, {"a": {"b": 2}})
        assert base == {"a": {"b": 1}}
