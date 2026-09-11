"""The `callbacks` section as the list a trainer is handed: names it may write, and one it may not."""

from __future__ import annotations

import pytest
from lightning.pytorch.callbacks import Callback, LearningRateMonitor

from src.callbacks.build import build_callbacks
from src.callbacks.registry import callback_registry
from src.config import ComponentConfig
from src.config.instantiate import resolve_factory


@pytest.mark.parametrize("name", sorted(callback_registry))
def test_every_registered_name_is_something_a_loop_can_call(name: str) -> None:
    resolved = resolve_factory(ComponentConfig(name=name), callback_registry)

    assert isinstance(resolved, type) and issubclass(resolved, Callback)


def test_the_declared_callbacks_arrive_in_the_order_they_were_written() -> None:
    """Order is a decision a run makes: one that changes the weights belongs before one that saves them."""
    built = build_callbacks(
        [
            ComponentConfig.model_validate({"name": "lr_monitor", "logging_interval": "epoch"}),
            ComponentConfig.model_validate({"name": "checkpoint", "monitor": "val/loss"}),
        ]
    )

    assert [type(one).__name__ for one in built] == ["LearningRateMonitor", "ModelCheckpoint"]
    assert isinstance(built[0], LearningRateMonitor) and built[0].logging_interval == "epoch"


def test_a_run_that_declares_none_gets_none() -> None:
    assert build_callbacks([]) == []


def test_something_with_no_hooks_to_run_is_refused_where_it_was_declared() -> None:
    with pytest.raises(TypeError, match="Counter"):
        build_callbacks([ComponentConfig.model_validate({"_target_": "collections.Counter"})])
