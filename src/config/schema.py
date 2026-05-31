"""Pydantic schema — the single validated contract for an experiment.

This is the boundary layer (per the project's dataclass-vs-Pydantic split):
YAML/Hydra produces a plain dict, which is validated here into typed models.
Nothing downstream re-parses raw config — services receive these DTOs.

Design notes:
- ``preset``/``objective``/backbone ``kind``/optimizer ``name`` stay free strings:
  their valid values live in the task/model/optim layers and are checked by the
  builders (with clear errors), so config does not couple to that taxonomy.
- Component sections allow ``extra`` keys so per-brick overrides and the
  ``_target_`` escape hatch survive validation for later wiring.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.core.enums import Stage

# ImageNet normalization — sensible defaults so simple configs omit them.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class DataConfig(BaseModel):
    """Where data comes from and how it is split."""

    source: str | list[str] = Field(..., description="Path(s) to a CSV/JSON annotation file.")
    image_column: str = Field(..., description="Column holding image paths.")
    split: dict[Stage, float] = Field(..., description="Train/val/test ratios; must sum to 1.0.")
    root_path: str | None = Field(None, description="Optional prefix prepended to image paths.")

    model_config = ConfigDict(extra="allow")

    @field_validator("split")
    @classmethod
    def _validate_split(cls, value: dict[Stage, float]) -> dict[Stage, float]:
        if Stage.PREDICT in value:
            raise ValueError("split may only contain train/val/test, not predict.")
        if Stage.TRAIN not in value:
            raise ValueError("split must include a 'train' ratio.")
        if any(ratio < 0 for ratio in value.values()):
            raise ValueError("split ratios must be non-negative.")
        total = sum(value.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"split ratios must sum to 1.0, got {total}.")
        return value


class BackboneConfig(BaseModel):
    """Backbone selection; ``kind`` picks the registry adapter."""

    kind: str = Field("timm", description="Backbone registry key (timm/smp/hf/embedding/...).")
    name: str = Field(..., description="Model name within the chosen backbone library.")
    pretrained: bool = Field(True, description="Load pretrained weights when supported.")

    model_config = ConfigDict(extra="allow")


class OptimizerConfig(BaseModel):
    """Optimizer selection and core hyper-parameters."""

    name: str = Field("adamw", description="Optimizer registry key.")
    lr: float = Field(..., gt=0, description="Learning rate.")
    weight_decay: float = Field(0.0, ge=0, description="Weight decay.")

    model_config = ConfigDict(extra="allow")


class TaskConfig(BaseModel):
    """One task declared under ``tasks`` (keyed by task name).

    ``preset`` selects a familiar task family (classification/segmentation/...);
    ``objective`` optionally overrides its label semantics. ``num_classes`` is
    omitted by default and inferred from data at runtime.
    """

    preset: str = Field(..., description="Task preset, e.g. 'classification'.")
    target: str = Field(..., description="Target column in the data source.")
    objective: str | None = Field(None, description="Override: binary/multiclass/multilabel/continuous.")
    num_classes: int | None = Field(None, gt=0, description="Class count; inferred from data if omitted.")
    dim: int | None = Field(None, gt=0, description="Output dimension for regression (replaces num_classes).")
    weight: float = Field(1.0, gt=0, description="Weight of this task in the total loss.")
    optimizer: OptimizerConfig | None = Field(None, description="Per-head optimizer override (own LR).")
    loss: str | dict[str, Any] | None = Field(
        None, description="Loss override: registry key, {name/_target_ + params}; None -> objective default."
    )
    metrics: dict[str, dict[str, Any] | None] | None = Field(
        None, description="Metric specs by label: {label: {params}}; None -> objective default."
    )
    target_codec: str | dict[str, Any] | None = Field(
        None, description="Data-codec override: registry key or {name/_target_ + params}; None -> inferred from objective."
    )

    model_config = ConfigDict(extra="allow")


class TrainerConfig(BaseModel):
    """Subset of Lightning Trainer knobs we expose; extras pass through."""

    accelerator: str = "auto"
    devices: int | str = "auto"
    precision: str = "32-true"
    log_every_n_steps: int = 10

    model_config = ConfigDict(extra="allow")


class ExperimentConfig(BaseModel):
    """Root experiment contract assembled from the YAML config."""

    project: str = Field(..., description="Project name for tracking.")
    seed: int = Field(42, description="Global random seed.")
    epochs: int = Field(..., gt=0, description="Number of training epochs.")
    batch_size: int = Field(..., gt=0, description="Batch size.")
    image_size: tuple[int, int] = Field(..., description="Image (height, width) in pixels.")
    mean: list[float] = Field(default_factory=lambda: list(_IMAGENET_MEAN), description="Normalization mean.")
    std: list[float] = Field(default_factory=lambda: list(_IMAGENET_STD), description="Normalization std.")
    data: DataConfig
    backbone: BackboneConfig
    optimizer: OptimizerConfig
    tasks: dict[str, TaskConfig] = Field(..., min_length=1, description="Tasks by name.")
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)

    model_config = ConfigDict(extra="forbid")

    @field_validator("image_size")
    @classmethod
    def _validate_image_size(cls, value: tuple[int, int]) -> tuple[int, int]:
        if any(side <= 0 for side in value):
            raise ValueError(f"image_size dimensions must be positive, got {value}.")
        return value

    @model_validator(mode="after")
    def _validate_normalization(self) -> ExperimentConfig:
        if len(self.mean) != len(self.std):
            raise ValueError(f"mean ({len(self.mean)}) and std ({len(self.std)}) must have equal length.")
        return self
