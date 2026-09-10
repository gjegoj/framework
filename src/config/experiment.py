"""Familiar experiment controls, independent of the model's input modalities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config.distillation import DistillationConfig
from src.config.schema import AdaptersConfig, ComponentConfig, ModelConfig, PreprocessingConfig, TaskConfig
from src.core import Stage, validate_name


def reject_owned_keys(values: Mapping[str, object], owned: Mapping[str, str]) -> None:
    for key, location in owned.items():
        if key in values:
            raise ValueError(f"Declare {key} once, at {location}.")


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: str | None = None
    name: str | None = None
    train: bool = True
    test: bool = True
    checkpoint_path: str | None = Field(None, min_length=1, description="Initial model weights; starts a fresh run.")
    resume_path: str | None = Field(None, min_length=1, description="Continue weights, optimizer, scheduler and epoch.")
    directory: str = Field("runs", min_length=1)

    @model_validator(mode="after")
    def restoration(self) -> RunConfig:
        if sum(path is not None for path in (self.checkpoint_path, self.resume_path)) > 1:
            raise ValueError("Choose one of run.checkpoint_path or run.resume_path.")
        if self.resume_path is not None and not self.train:
            raise ValueError("run.resume_path requires run.train=true.")
        return self


class ForwardSection(BaseModel):
    """Declared fields are validated; every other key reaches the library constructor verbatim.

    Keys the framework owns elsewhere are refused by name, so a value is declared once.
    """

    model_config = ConfigDict(extra="allow")
    owned: ClassVar[Mapping[str, str]] = {}

    @model_validator(mode="after")
    def framework_arguments(self) -> Self:
        reject_owned_keys(self.params, self.owned)
        return self

    @property
    def params(self) -> dict[str, object]:
        """The forwarded keys alone; declared fields are read as attributes."""
        return dict(self.model_extra or {})


class LoaderConfig(ForwardSection):
    owned: ClassVar[Mapping[str, str]] = {
        "batch_size": "experiment.batch_size",
        "dataset": "data",
        "collate_fn": "preprocessing.collator",
        "shuffle": "the stage: training shuffles, evaluation preserves order",
        "batch_sampler": "a custom DataLoader adapter (it replaces fixed batch_size)",
    }

    num_workers: int = Field(0, ge=0)
    pin_memory: bool = False
    drop_last: bool = Field(False, description="Training only; evaluation preserves every sample.")


class TrainerConfig(ForwardSection):
    owned: ClassVar[Mapping[str, str]] = {
        "max_epochs": "experiment.epochs",
        "logger": "experiment.tracker",
        "callbacks": "experiment.callbacks",
        "default_root_dir": "run.directory",
    }

    accelerator: str = "auto"
    devices: int | str | list[int] = "auto"
    profiler: ComponentConfig | None = None


class SchedulerConfig(ComponentConfig):
    interval: Literal["epoch", "step"] = "epoch"
    frequency: int = Field(1, gt=0)
    monitor: str | None = None
    strict: bool = True


class ExperimentConfig(BaseModel):
    """Assembly injects root controls; runtime objects never read this schema.

    Preprocessing owns modality-specific loading, normalization and collation.
    Its component may compose named inputs or wrap one joint image/text processor.
    """

    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    lr: float = Field(1e-3, gt=0, allow_inf_nan=False)
    batch_size: int = Field(16, gt=0, strict=True)
    epochs: int = Field(10, gt=0, strict=True)
    data: ComponentConfig
    preprocessing: PreprocessingConfig | None = Field(None, description="None requires already prepared model inputs.")
    transforms: dict[Stage, ComponentConfig] = Field(
        default_factory=dict, description="Stage-specific sample augmentation, before final normalization and encoding."
    )
    model: ModelConfig
    tasks: dict[str, TaskConfig]
    adapters: AdaptersConfig = Field(default_factory=list)
    distillation: DistillationConfig | None = None
    learner: ComponentConfig = Field(default_factory=lambda: ComponentConfig(name="standard"))
    optimizer: ComponentConfig = Field(default_factory=lambda: ComponentConfig(name="adamw"))
    scheduler: SchedulerConfig | None = None
    loader: LoaderConfig = Field(default_factory=LoaderConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    callbacks: list[ComponentConfig] = Field(default_factory=list)
    tracker: ComponentConfig | None = None
    export: list[ComponentConfig] = Field(default_factory=list)
    run: RunConfig = Field(default_factory=RunConfig)

    @model_validator(mode="after")
    def connections(self) -> ExperimentConfig:
        reject_owned_keys(self.optimizer.params, {"lr": "experiment.lr"})
        if not self.tasks:
            raise ValueError("An experiment requires at least one task.")
        for name in self.tasks:
            validate_name(name, kind="Task")
        if "heads" in self.model.params:
            raise ValueError("Declare heads once, under tasks; the model section only selects the network.")
        return self
