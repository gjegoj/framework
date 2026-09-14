"""Validated declarations; importing schemas does not resolve implementations.

The resolver is one import further in, at ``src.config.instantiate``, and deliberately not re-exported
here: a declaration and the construction of what it names are two things, and a builder that does both
says so in two lines. Reading a schema stays free of everything a ``_target_`` could reach.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.config.experiment import (
    ExperimentConfig,
    LoaderConfig,
    RunConfig,
    SchedulerConfig,
    TeacherConfig,
    TrainerConfig,
)
from src.config.schema import (
    ClassFile,
    ComponentConfig,
    HeadConfig,
    ModelConfig,
    PreprocessingConfig,
    TaskConfig,
    WeightedLossConfig,
)


def load_config(raw: Mapping[str, Any]) -> ExperimentConfig:
    """The config boundary: a composed, resolved mapping in, one validated experiment out."""
    return ExperimentConfig.model_validate(raw)


__all__ = [
    "ClassFile",
    "ComponentConfig",
    "ExperimentConfig",
    "HeadConfig",
    "LoaderConfig",
    "ModelConfig",
    "PreprocessingConfig",
    "RunConfig",
    "SchedulerConfig",
    "TaskConfig",
    "TeacherConfig",
    "TrainerConfig",
    "WeightedLossConfig",
    "load_config",
]
