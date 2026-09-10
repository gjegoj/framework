"""Validated declarations; importing schemas does not resolve implementations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.config.distillation import DistillationConfig, TeacherConfig
from src.config.experiment import ExperimentConfig, LoaderConfig, RunConfig, SchedulerConfig, TrainerConfig
from src.config.schema import (
    AdapterConfig,
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
    "AdapterConfig",
    "ClassFile",
    "ComponentConfig",
    "DistillationConfig",
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
