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
    """Where data comes from and how it is divided into stages.

    The ``sources`` field drives both modes — its type determines which:

    **Split mode** — string or list of strings, ratios decide the split::

        data:
          sources: data/annotations.csv
          split: {train: 0.8, val: 0.1, test: 0.1}
          inputs: image_path                        # shorthand: one image input

    **Pre-split mode** — dict keyed by stage::

        data:
          sources:
            train: [data/train_a.csv, data/train_b.csv]
            val: data/val.csv
          inputs: image_path

    **Multiple inputs** (multi-view / multimodal)::

        data:
          inputs:
            left_image: left_path          # loader auto-detected from extension
            right_image: right_path
          # explicit loader:
          # inputs:
          #   image: image_path
          #   caption: {column: text_col, loader: text}

    ``source_type`` (csv/json) is inferred from the file extension when omitted.
    """

    sources: str | list[str] | dict[str, str | list[str]] = Field(
        ...,
        description=(
            "Annotation path(s) or per-stage dict. "
            "str/list[str] → split mode (requires 'split'). "
            "dict[stage, paths] → pre-split mode ('train'/'val'/'test' keys)."
        ),
    )
    split: dict[Stage, float] | None = Field(None, description="Train/val/test ratios summing to 1.0 (split mode).")
    split_stratify: str | None = Field(
        None,
        description=(
            "Column to stratify by when splitting (split mode only). "
            "Auto-detected strategy: categorical strings → classification, "
            "numeric → quantile-binned, comma-separated strings → multilabel "
            "(IterativeStratification)."
        ),
    )
    max_samples: int | float | None = Field(
        None,
        gt=0,
        description=(
            "Cap dataset size for fast iteration or debugging. "
            "int → keep exactly N rows; float in (0, 1] → keep this fraction. "
            "In split mode applied before splitting (caps total). "
            "In pre-split mode applied per stage."
        ),
    )
    inputs: str | dict[str, str | dict[str, str]] = Field(
        ...,
        description=(
            "Input column(s) and their loaders. "
            "str shorthand → single image input named 'image'. "
            "dict[alias, column] → multiple inputs, loader auto-detected from values. "
            "dict[alias, {column, loader}] → explicit loader key."
        ),
    )
    source_type: str | None = Field(None, description="data_sources key (csv/json); None -> inferred from extension.")
    root_path: str | None = Field(None, description="Optional prefix prepended to file-based input paths.")

    model_config = ConfigDict(extra="allow")

    @field_validator("split")
    @classmethod
    def _validate_split(cls, value: dict[Stage, float] | None) -> dict[Stage, float] | None:
        if value is None:
            return value
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

    @field_validator("max_samples")
    @classmethod
    def _validate_max_samples(cls, value: int | float | None) -> int | float | None:
        if isinstance(value, float) and not (0.0 < value <= 1.0):
            raise ValueError(f"max_samples as a fraction must be in (0, 1], got {value}.")
        return value

    @model_validator(mode="after")
    def _validate_mode(self) -> DataConfig:
        is_presplit = isinstance(self.sources, dict)
        if is_presplit:
            assert isinstance(self.sources, dict)
            valid = {"train", "val", "test"}
            invalid = set(self.sources) - valid
            if invalid:
                raise ValueError(f"sources keys must be in {{train, val, test}}, got: {sorted(invalid)}.")
            if "train" not in self.sources:
                raise ValueError("sources dict must include 'train'.")
            if self.split is not None:
                raise ValueError("'split' cannot be used when sources is a dict (pre-split mode).")
        else:
            if self.split is None:
                raise ValueError("'split' is required when sources is a path (split mode).")
        return self


class DataLoaderConfig(BaseModel):
    """DataLoader knobs shared across all stages.

    ``shuffle`` and ``drop_last`` for the train stage are conventions, not config:
    train always shuffles and optionally drops the last incomplete batch;
    val/test never shuffle and never drop.  Only ``drop_last`` is exposed here
    because it occasionally needs to be disabled (e.g. when dataset size is
    exactly divisible and the last batch matters for metrics).
    """

    num_workers: int = Field(0, ge=0, description="Worker processes per DataLoader. 0 → main process (debug-friendly).")
    pin_memory: bool = Field(
        False, description="Pin host memory for faster CPU→GPU transfers. Enable when training on GPU."
    )
    persistent_workers: bool = Field(
        False, description="Keep worker processes alive between epochs. Requires num_workers > 0."
    )
    drop_last: bool = Field(
        False, description="Drop the last incomplete batch during training (val/test are never dropped)."
    )
    prefetch_factor: int | None = Field(
        None, ge=1, description="Batches prefetched per worker. None → PyTorch default (2). Requires num_workers > 0."
    )

    model_config = ConfigDict(extra="forbid")


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
    head: str | dict[str, Any] | None = Field(
        None,
        description=(
            "Head override. ``None`` → backbone-native head (default). "
            "``str`` → registry key (e.g. 'conv'). "
            "``dict`` → ``{kind, ...options}`` or ``{_target_: my.Head, ...}``."
        ),
    )
    feature_key: str | None = Field(
        None,
        description=(
            "Which backbone stream this task's head consumes. "
            "``None`` → topology default (``pooled`` for classification, ``decoder`` for segmentation). "
            "Set explicitly for multitask on an encoder-decoder backbone, e.g.: "
            "``feature_key: encoder_last`` to use smp's ClassificationHead (has adaptive-avg-pool inside). "
            "Available streams are backbone-specific — see the backbone's docstring for the ``Streams`` table. "
            "An unknown key raises a ``KeyError`` listing what the backbone actually exposes."
        ),
    )
    class_mapping: dict[int, str] | None = Field(
        None,
        description=(
            "Explicit index→label mapping for categorical targets. "
            "Required for 'classification' and 'multilabel' presets; "
            "determines class count and index ordering. "
            "Example: {0: 'cat', 1: 'dog', 2: 'cow'}."
        ),
    )
    num_classes: int | None = Field(None, gt=0, description="Class count; inferred from class_mapping when omitted.")
    dim: int | None = Field(
        None,
        gt=0,
        description="Output dimension for regression (replaces num_classes).",
    )
    weight: float = Field(1.0, gt=0, description="Weight of this task in the total loss.")
    optimizer: OptimizerConfig | None = Field(None, description="Per-head optimizer override (own LR).")
    loss: str | dict[str, Any] | None = Field(
        None,
        description="Loss override: registry key, {name/_target_ + params}; None -> objective default.",
    )
    metrics: dict[str, dict[str, Any] | None] | None = Field(
        None,
        description="Metric specs by label: {label: {params}}; None -> objective default.",
    )
    target_codec: str | dict[str, Any] | None = Field(
        None,
        description="Data-codec override: registry key or {name/_target_ + params}; None -> inferred from objective.",
    )

    model_config = ConfigDict(extra="allow")


class LoggerConfig(BaseModel):
    """Logger backend selection.

    ``kind`` selects the backend (none/clearml); extras survive validation so
    per-backend YAML keys (``tags``, ``output_uri``, ...) can be forwarded.

    Example (clearml)::

        logger:
          kind: clearml
          project: my-ml-project   # defaults to experiment project when omitted
          task: run-001            # optional ClearML task name
    """

    kind: str = Field("none", description="Logger backend key (none/clearml).")
    project: str | None = Field(None, description="Project name for the logger. Defaults to experiment project.")
    task: str | None = Field(None, description="Run/task name. Logger backend default when omitted.")

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
    lr: float = Field(
        ...,
        gt=0,
        description="Global learning rate — referenced by optimizer.lr via ${lr}. Override per experiment or per task via tasks.<name>.optimizer.lr.",
    )
    data: DataConfig
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    backbone: BackboneConfig
    optimizer: OptimizerConfig
    tasks: dict[str, TaskConfig] = Field(..., min_length=1, description="Tasks by name.")
    transforms: dict[str, Any] | None = Field(
        None,
        description=(
            "Per-stage Albumentations pipeline specs (train/val/test). "
            "Each value is a _target_-keyed dict instantiated via instantiate_nested. "
            "None → default resize + normalize + ToTensorV2 built from image_size/mean/std."
        ),
    )
    logger: LoggerConfig = Field(default_factory=LoggerConfig, description="Logger backend config.")
    callbacks: dict[str, dict[str, Any] | None] | None = Field(
        None,
        description=(
            "Callbacks by registry key (or ``_target_``). "
            "Keys are looked up in ``callback_registry``; values are constructor kwargs. "
            "``null`` value → callback with all defaults. "
            "``_target_`` key → full import path bypass (no registry needed). "
            "YAML order controls registration order — put ``ema`` before ``checkpoint``. "
            "Remove a key (or set to ``~``) to disable that callback. "
            "Example: ``{lr_monitor: {logging_interval: epoch}, ema: {decay: 0.999}, checkpoint: null}``."
        ),
    )
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
