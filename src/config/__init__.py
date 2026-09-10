"""Validated declarations; importing schemas does not resolve implementations."""

from __future__ import annotations

from src.config.distillation import DistillationConfig, TeacherConfig
from src.config.experiment import ExperimentConfig, LoaderConfig, RunConfig, SchedulerConfig, TrainerConfig
from src.config.schema import (
    AdapterConfig,
    ClassFile,
    ComponentConfig,
    HeadConfig,
    TaskConfig,
    WeightedLossConfig,
)

__all__ = [
    "AdapterConfig",
    "ClassFile",
    "ComponentConfig",
    "DistillationConfig",
    "ExperimentConfig",
    "HeadConfig",
    "LoaderConfig",
    "RunConfig",
    "SchedulerConfig",
    "TaskConfig",
    "TeacherConfig",
    "TrainerConfig",
    "WeightedLossConfig",
]
