"""Every registered callback constructs from its documented required arguments, and is a Lightning callback."""

from __future__ import annotations

from typing import Any

import lightning as L
import pytest

from src.callbacks.registry import callback_registry
from src.transforms import MixUp

REQUIRED: dict[str, dict[str, Any]] = {
    "lr_monitor": {},
    "checkpoint": {},
    "anneal": {"task": "label", "parameter": "label_smoothing", "start": 0.1, "end": 0.0},
    "freeze": {"modules": ["backbone"]},
    "batch_transform": {"transform": MixUp()},
    "ema": {},
    "ema_checkpoint": {},
    "dataset_summary": {},
    "metric_summary": {},
    "model_summary": {},
    "progress": {},
    "samples": {
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    },  # the run's normalisation, never a default of the page's
}
"""Name → what a declaration has to carry; the rest has a default."""


def test_every_registered_callback_has_a_row_here() -> None:
    assert set(map(str, callback_registry)) == set(REQUIRED)


@pytest.mark.parametrize("name", sorted(REQUIRED))
def test_it_constructs_from_its_required_arguments_alone(name: str) -> None:
    assert isinstance(callback_registry.create(name, **REQUIRED[name]), L.Callback)
