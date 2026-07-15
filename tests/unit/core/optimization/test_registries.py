"""Тесты реестров оптимизаторов и шедулеров."""

from torch import optim

from perseus.core.optimization import optimizers, schedulers


def test_optimizer_registry_maps_to_optim_classes() -> None:
    assert optimizers.registry["adamw"] is optim.AdamW
    assert all(issubclass(cls, optim.Optimizer) for cls in optimizers.registry.values())


def test_scheduler_registry_has_expected_keys() -> None:
    expected = {"constant", "linear", "cosine", "polynomial", "inverse_sqrt"}
    assert expected <= set(schedulers.registry)
    assert all(callable(factory) for factory in schedulers.registry.values())
