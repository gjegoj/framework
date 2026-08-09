"""The config boundary: raw mappings in, one typed contract out."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.config.components import ComponentConfig, MetricConfig, ModelConfig, TransformConfig
from src.config.data import CacheConfig, DataConfig, InputColumnConfig, InputLoaderConfig, SplitConfig
from src.config.distillation import DistillationConfig, TeacherConfig
from src.config.experiment import CallbackConfig, ExperimentConfig, LoggerConfig
from src.config.presets import TaskPreset, resolve_preset
from src.config.registry import task_preset_registry
from src.config.run import RunConfig
from src.config.tasks import HeadConfig, LossConfig, TargetEncoderConfig, TaskConfig
from src.config.training import LoaderConfig, OptimizerConfig, SchedulerConfig, TrainerConfig


def load_config(raw: Mapping[str, Any]) -> ExperimentConfig:
    """Validate a raw mapping into the typed experiment contract.

    The single entry point of the config boundary, and the only place validation
    happens. Pydantic lives inside this package alone: capability layers receive plain
    typed objects and never parse raw config. Composition — Hydra groups, CLI
    overrides — is the composition root's business; this package owns validation.

    Parameters:
        raw (Mapping[str, Any]): The composed config, with every interpolation
            already resolved.

    Raises:
        pydantic.ValidationError: On any structural or semantic violation,
            naming the offending section and key.
    """
    return ExperimentConfig.model_validate(raw)


__all__ = [
    "CacheConfig",
    "CallbackConfig",
    "ComponentConfig",
    "DataConfig",
    "DistillationConfig",
    "ExperimentConfig",
    "HeadConfig",
    "InputColumnConfig",
    "InputLoaderConfig",
    "LoaderConfig",
    "LoggerConfig",
    "LossConfig",
    "MetricConfig",
    "ModelConfig",
    "OptimizerConfig",
    "RunConfig",
    "SchedulerConfig",
    "SplitConfig",
    "TargetEncoderConfig",
    "TaskConfig",
    "TaskPreset",
    "TeacherConfig",
    "TrainerConfig",
    "TransformConfig",
    "load_config",
    "resolve_preset",
    "task_preset_registry",
]
