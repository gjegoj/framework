"""Validated declarations built from a few overrides, so a test names only what it is about."""

from __future__ import annotations

from typing import Any

from src.config import ComponentConfig, PreprocessingConfig, TaskConfig

CLASSES = {0: "cat", 1: "dog"}


def task_config(**declared: Any) -> TaskConfig:
    """A classification task on the ``species`` column unless a key says otherwise."""
    return TaskConfig.model_validate({"kind": "classification", "target": "species", **declared})


def component(declared: Any) -> ComponentConfig:
    return ComponentConfig.model_validate(declared)


def preprocessing_config(**declared: Any) -> PreprocessingConfig:
    """The standard image preprocessor at 4x4, plus whatever a test adds."""
    return PreprocessingConfig.model_validate(
        {"name": "standard", "inputs": {"image": {"name": "image", "image_size": [4, 4]}}, **declared}
    )
