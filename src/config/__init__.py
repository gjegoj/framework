"""Config: the validated Pydantic contract for an experiment.

The composition root converts a Hydra ``DictConfig`` to a plain dict and calls
:func:`load_config`; everything downstream consumes typed DTOs only.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from src.config.schema import (
    BackboneConfig,
    DataConfig,
    ExperimentConfig,
    ExportConfig,
    OptimizerConfig,
    TaskConfig,
    TrainerConfig,
)


class ConfigError(ValueError):
    """Raised when a raw config fails validation against the schema."""


def load_config(raw: dict[str, Any]) -> ExperimentConfig:
    """Validate a raw config mapping into a typed :class:`ExperimentConfig`.

    Parameters:
        raw (dict[str, Any]): Plain mapping (e.g. from ``OmegaConf.to_container``).

    Returns:
        ExperimentConfig: The validated experiment contract.

    Raises:
        ConfigError: If validation fails, with the underlying Pydantic report.
    """
    try:
        return ExperimentConfig.model_validate(raw)
    except ValidationError as error:
        raise ConfigError(f"Invalid experiment config:\n{error}") from error


__all__ = [
    "BackboneConfig",
    "ConfigError",
    "DataConfig",
    "ExperimentConfig",
    "ExportConfig",
    "OptimizerConfig",
    "TaskConfig",
    "TrainerConfig",
    "load_config",
]
