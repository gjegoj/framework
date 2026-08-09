"""The callback registry: what a config may name, and what it gets."""

from __future__ import annotations

import pytest
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from src.callbacks.registry import callback_registry


def test_lightning_callbacks_are_registered_not_wrapped() -> None:
    """Wrapping would add a translation layer over an interface that already fits."""
    assert callback_registry.get("lr_monitor") is LearningRateMonitor
    assert callback_registry.get("checkpoint") is ModelCheckpoint


def test_a_registered_callback_is_built_with_its_arguments() -> None:
    built = callback_registry.create("checkpoint", monitor="val/loss", mode="min")

    assert isinstance(built, ModelCheckpoint)
    assert built.monitor == "val/loss"
    assert built.mode == "min"


def test_a_checkpoint_leaves_its_directory_to_lightning() -> None:
    """Lightning resolves it from the logger's log_dir, then default_root_dir."""
    built = callback_registry.create("checkpoint", monitor="val/loss")

    assert isinstance(built, ModelCheckpoint)
    assert built.dirpath is None


def test_an_unknown_name_lists_the_registered_ones() -> None:
    with pytest.raises(LookupError, match="checkpoint"):
        callback_registry.create("nope")
