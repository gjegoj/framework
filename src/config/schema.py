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
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.config.export import ExportConfig
from src.core.constants import IMAGENET_MEAN, IMAGENET_STD
from src.core.enums import Stage


class CacheConfig(BaseModel):
    """In-RAM cache of decoded images/masks, warmed in the parent before training.

    Budget is ``min(ram_fraction * available_RAM, max_gb)``. Caching is read-only
    after warm-up, so cached buffers stay shared across DataLoader fork workers.
    """

    ram_fraction: float = Field(0.5, ge=0.0, le=1.0, description="Cap as a fraction of available RAM (0 disables).")
    max_gb: float | None = Field(None, gt=0, description="Absolute cap in GiB; budget = min(fraction·RAM, max_gb).")
    workers: int = Field(8, ge=1, description="Threads used to warm the cache.")

    model_config = ConfigDict(extra="forbid")


class SourceConfig(BaseModel):
    """One source in a ``data.sources`` list: path(s) plus optional per-stage transforms.

    Gives a source its own augmentation pipeline. Override is **replace** semantics: a
    per-stage override fully replaces the global stage transform for this source's rows;
    an unset stage falls back to the global ``transforms`` group. A plain string in the
    ``sources`` list is shorthand for ``SourceConfig(path=...)`` with no override.

    Parameters:
        path (str | list[str]): Annotation file(s) for this source.
        transforms (dict[str, Any] | None): Per-stage transform spec
            (``{train|val|test: <_target_ spec>}``); ``None`` → use the global ``transforms``.
    """

    path: str | list[str] = Field(..., description="Annotation file(s) for this source.")
    transforms: dict[str, Any] | None = Field(
        None,
        description=(
            "Per-source transform override (replace semantics; a per-source pipeline must itself end "
            "in Normalize + ToTensorV2). Two forms by mode: in split mode a per-stage dict "
            "{train|val|test: spec} (the source spans all stages); in pre-split mode a single transform "
            "spec {_target_: ...} (the source is already pinned to its stage). None → global transforms."
        ),
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("transforms")
    @classmethod
    def _validate_transforms_form(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        """Reject anything that is neither a single transform spec nor a clean per-stage dict."""
        if value is None:
            return value
        stage_names = {stage.value for stage in Stage}
        has_target = "_target_" in value
        if has_target and not any(key in stage_names for key in value):
            return value  # single transform spec
        if value and not has_target and all(key in stage_names for key in value):
            return value  # per-stage dict
        raise ValueError(
            "source 'transforms' must be EITHER a single transform spec ({_target_: ...}) OR a per-stage "
            f"dict ({{train|val|test: spec}}); got keys {sorted(value)} (mixed / unknown keys are not allowed)."
        )

    @property
    def is_single_transform(self) -> bool:
        """True when ``transforms`` is a single transform spec (``_target_``), not a per-stage dict."""
        return self.transforms is not None and "_target_" in self.transforms


def _source_configs(value: object) -> list[SourceConfig]:
    """The ``SourceConfig`` entries within one ``sources`` value (string / ``SourceConfig`` / list)."""
    if isinstance(value, SourceConfig):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, SourceConfig)]
    return []


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

    sources: str | list[str | SourceConfig] | dict[str, str | SourceConfig | list[str | SourceConfig]] = Field(
        ...,
        description=(
            "Annotation path(s) or per-stage dict. "
            "str/list[str] → split mode (requires 'split'). "
            "A list item may be a SourceConfig ({path, transforms}) to give that source its own "
            "augmentations — a per-stage dict in split mode, a single transform under a stage in pre-split. "
            "dict[stage, paths|SourceConfig|list] → pre-split mode ('train'/'val'/'test' keys)."
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
    cache: CacheConfig | None = Field(None, description="In-RAM image/mask cache; None disables.")

    # Closed field set (no per-brick overrides / _target_ escape hatch here, unlike the component
    # sections), so forbid extras — a typo like `split_stratifi` fails loudly instead of being swallowed.
    model_config = ConfigDict(extra="forbid")

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
        if isinstance(self.sources, dict):
            valid = {"train", "val", "test"}
            invalid = set(self.sources) - valid
            if invalid:
                raise ValueError(f"sources keys must be in {{train, val, test}}, got: {sorted(invalid)}.")
            if "train" not in self.sources:
                raise ValueError("sources dict must include 'train'.")
            if self.split is not None:
                raise ValueError("'split' cannot be used when sources is a dict (pre-split mode).")
            # A pre-split source is already pinned to its stage → its transform must be a single spec.
            for stage_key, value in self.sources.items():
                for source in _source_configs(value):
                    if source.transforms is not None and not source.is_single_transform:
                        raise ValueError(
                            f"pre-split source under '{stage_key}' must use a single transform spec "
                            "({_target_: ...}), not a per-stage dict — it is already pinned to this stage."
                        )
        else:
            if self.split is None:
                raise ValueError("'split' is required when sources is a path (split mode).")
            # A split-mode source spans every stage → its transform must be a per-stage dict.
            for source in _source_configs(self.sources):
                if source.transforms is not None and source.is_single_transform:
                    raise ValueError(
                        "split-mode source 'transforms' must be a per-stage dict ({train|val|test: spec}), "
                        "not a single transform — the source spans all stages."
                    )
        return self


class DataLoaderConfig(BaseModel):
    """DataLoader knobs shared across all stages.

    ``shuffle`` and ``drop_last`` for the train stage are conventions, not config:
    train always shuffles and optionally drops the last incomplete batch;
    val/test never shuffle and never drop.  Only ``drop_last`` is exposed here
    because it occasionally needs to be disabled (e.g. when dataset size is
    exactly divisible and the last batch matters for metrics).

    Extra keys are allowed and forwarded verbatim to ``torch.utils.data.DataLoader``
    (e.g. ``timeout``, ``multiprocessing_context``, ``pin_memory_device``), mirroring
    how ``trainer``/``optimizer`` forward their extras. Keys the framework owns
    (``RESERVED``: dataset/batch_size/shuffle/collate_fn/sampler/batch_sampler) are
    rejected so per-stage conventions cannot be silently broken.
    """

    RESERVED: ClassVar[frozenset[str]] = frozenset(
        {"dataset", "batch_size", "shuffle", "collate_fn", "sampler", "batch_sampler"}
    )

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

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _reject_reserved_extras(self) -> DataLoaderConfig:
        clashing = self.RESERVED & set(self.model_extra or {})
        if clashing:
            raise ValueError(
                f"dataloader keys {sorted(clashing)} are managed by the framework "
                "(per-stage shuffle/drop_last, batch_size, collate_fn) and cannot be set here."
            )
        return self


class BackboneConfig(BaseModel):
    """Backbone selection; ``kind`` picks the registry adapter."""

    kind: str = Field("timm", description="Backbone registry key (timm/smp/hf/embedding/multi/...).")
    name: str | None = Field(
        None,
        description="Model name within the chosen backbone library. None for composite backbones (kind=multi).",
    )
    pretrained: bool = Field(True, description="Load pretrained weights when supported.")

    model_config = ConfigDict(extra="allow")


class OptimizerConfig(BaseModel):
    """Optimizer selection and core hyper-parameters."""

    name: str = Field("adamw", description="Optimizer registry key.")
    lr: float = Field(..., gt=0, description="Learning rate.")
    weight_decay: float = Field(0.0, ge=0, description="Weight decay.")

    model_config = ConfigDict(extra="allow")


class SchedulerConfig(BaseModel):
    """LR scheduler selection, Lightning scheduling policy, and scheduler kwargs.

    Mirrors ``OptimizerConfig``: ``name`` selects a class from the ``schedulers``
    registry; the policy fields map to Lightning's ``lr_scheduler`` config; all
    other fields (``T_max``, ``max_lr``, ``factor``, …) forward verbatim to the
    constructor. ``runtime_kwargs`` maps a constructor parameter to a trainer
    fact (``total_steps`` / ``steps_per_epoch`` / ``epochs``) filled at fit time.
    """

    name: str = Field(..., description="Scheduler registry key (cosine / onecycle / plateau / step).")
    interval: Literal["epoch", "step"] = Field("epoch", description="When Lightning steps the scheduler.")
    frequency: int = Field(1, ge=1, description="Step the scheduler every N intervals.")
    monitor: str | None = Field(None, description="Metric to monitor (required for ReduceLROnPlateau).")
    strict: bool = Field(True, description="Error if the monitored metric is missing.")
    runtime_kwargs: dict[str, str] = Field(
        default_factory=dict,
        description="Constructor param → trainer fact (total_steps / steps_per_epoch / epochs).",
    )

    model_config = ConfigDict(extra="allow")


class TaskConfig(BaseModel):
    """One task declared under ``tasks`` (keyed by task name).

    ``preset`` selects a familiar task family (classification/segmentation/...);
    ``objective`` optionally overrides its label semantics. ``num_classes`` is
    omitted by default and inferred from data at runtime.
    """

    preset: str = Field(..., description="Task preset, e.g. 'classification'.")
    target: str | None = Field(
        None,
        description=(
            "Target column in the data source. Omit (``None``) for a target-less task "
            "(triplet / contrastive) supervised purely by structure — no column required; "
            "the wiring then uses the ``null`` target encoder."
        ),
    )
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
    target_encoder: str | dict[str, Any] | None = Field(
        None,
        description="Data-encoder override: registry key or {name/_target_ + params}; None -> inferred from objective.",
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
    tags: list[str] | None = Field(
        None,
        description=(
            "Backend task tags (ClearML). Supports ${...} interpolation, e.g. "
            "'lr=${lr}' or '${optimizer.name}'. A tag that resolves to null (e.g. "
            "${backbone.name} on a composite backbone) or empty is dropped."
        ),
    )

    model_config = ConfigDict(extra="allow")

    @field_validator("tags", mode="before")
    @classmethod
    def _drop_empty_tags(cls, value: list[Any] | None) -> list[str] | None:
        """Drop null / empty tags (e.g. an unresolved ${backbone.name}) and stringify the rest."""
        if value is None:
            return None
        return [str(tag) for tag in value if tag is not None and str(tag) != ""]


class TrainerConfig(BaseModel):
    """Subset of Lightning Trainer knobs we expose; extras pass through."""

    accelerator: str = "auto"
    devices: int | str = "auto"
    precision: str = "32-true"
    log_every_n_steps: int = 10
    profiler: str | dict[str, Any] | None = Field(
        None,
        description=(
            "Lightning profiler. String alias ('simple'/'advanced'/'pytorch') is passed through; "
            "a {_target_: ...} mapping is built via the _target_ grammar (e.g. AdvancedProfiler with "
            "dirpath/filename to write the report to a file). None disables profiling."
        ),
    )

    model_config = ConfigDict(extra="allow")


class ExperimentConfig(BaseModel):
    """Root experiment contract assembled from the YAML config."""

    project: str = Field(..., description="Project name for tracking.")
    run_name: str | None = Field(
        None,
        description=(
            "Human-readable name for this run. "
            "Flows to the logger as the task/run name. "
            "Tip: set to '${now:%Y-%m-%d_%H-%M-%S}' for an auto-timestamp."
        ),
    )
    save_dir: str | None = Field(
        None,
        description=(
            "Root directory for run outputs (checkpoints, logs). "
            "When set, checkpoint.dirpath defaults to '{save_dir}/checkpoints' "
            "and Trainer.default_root_dir is set to this path. "
            "Tip: set to '${hydra:run.dir}' to use Hydra's run directory."
        ),
    )
    seed: int = Field(42, description="Global random seed.")
    run_train: bool = Field(
        True,
        description="Run ``trainer.fit()``. Set ``false`` for checkpoint-only evaluation.",
    )
    run_test: bool = Field(
        True,
        description=(
            "Run ``trainer.test()`` on the test split. After training, uses the best "
            "saved checkpoint when available; otherwise in-memory weights. Requires "
            "``ckpt_path`` when ``run_train`` is ``false``."
        ),
    )
    run_export: bool = Field(
        True,
        description="Export the model after train/test using ``export`` settings.",
    )
    ckpt_path: str | None = Field(
        None,
        description=(
            "Checkpoint for ``trainer.test()`` — absolute/relative ``.ckpt`` path, or "
            "Lightning aliases ``best`` / ``last``. Required for eval-only "
            "(``run_train: false``). When omitted after training, ``best`` is used if "
            "``ModelCheckpoint`` saved one. Not used to initialize training — see "
            "``init_ckpt_path``."
        ),
    )
    init_ckpt_path: str | None = Field(
        None,
        description=(
            "Pretrain / fine-tune: load model weights from this ``.ckpt`` into the "
            "module before ``fit``. Optimizer, schedulers, and epoch counter start "
            "fresh (not resume). Uses the checkpoint ``state_dict`` (EMA weights when "
            "the file was saved with EMA). Requires ``run_train: true``."
        ),
    )
    epochs: int = Field(..., gt=0, description="Number of training epochs.")
    batch_size: int = Field(..., gt=0, description="Batch size.")
    image_size: tuple[int, int] = Field(..., description="Image (height, width) in pixels.")
    mean: list[float] = Field(default_factory=lambda: list(IMAGENET_MEAN), description="Normalization mean.")
    std: list[float] = Field(default_factory=lambda: list(IMAGENET_STD), description="Normalization std.")
    lr: float = Field(
        ...,
        gt=0,
        description=(
            "Global learning rate — referenced by optimizer.lr via ${lr}. "
            "Override per experiment or per task via tasks.<name>.optimizer.lr."
        ),
    )
    data: DataConfig
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    backbone: BackboneConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig | None = Field(None, description="LR scheduler config; None = constant LR.")
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
    export: ExportConfig = Field(default_factory=ExportConfig, description="Model export settings.")

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

    @model_validator(mode="after")
    def _validate_run_modes(self) -> ExperimentConfig:
        if not self.run_train and not self.run_test and not self.run_export:
            raise ValueError("At least one of run_train, run_test, or run_export must be True.")
        needs_ckpt = not self.run_train and (self.run_test or self.run_export)
        if needs_ckpt and self.ckpt_path is None:
            raise ValueError("ckpt_path is required when run_train=false and run_test or run_export is true.")
        if not self.run_train and self.init_ckpt_path is not None:
            raise ValueError("init_ckpt_path requires run_train=true; use ckpt_path for eval-only runs.")
        return self
