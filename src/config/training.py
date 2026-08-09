"""The training sections: loader, optimizer, scheduler and trainer knobs."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config.components import ComponentConfig


class LoaderConfig(BaseModel):
    """DataLoader knobs shared by every stage; unknown keys forward to torch.

    ``shuffle`` and ``drop_last`` are stage conventions rather than free
    settings — training shuffles and may drop its last incomplete batch, while
    evaluation does neither, because a dropped batch means metrics computed on
    part of the split. ``drop_last`` is therefore declared here (it is a real
    choice for training) while the arguments the adapter passes itself are
    rejected instead of colliding at call time.
    """

    model_config = ConfigDict(extra="allow")

    ADAPTER_OWNED: ClassVar[frozenset[str]] = frozenset({"dataset", "shuffle", "collate_fn"})

    batch_size: int = Field(16, gt=0, description="Samples per batch.")
    num_workers: int = Field(0, ge=0, description="Worker processes loading samples; 0 loads in the main process.")
    pin_memory: bool = Field(False, description="Pin host memory to speed up transfers to a GPU.")
    drop_last: bool = Field(
        False,
        description=(
            "Drop the last incomplete training batch. Training only: dropping it during evaluation "
            "would compute metrics on part of the split."
        ),
    )

    @model_validator(mode="after")
    def _reject_adapter_owned_keys(self) -> LoaderConfig:
        """Refused here by name, rather than colliding inside the DataLoader call."""
        clashing = self.ADAPTER_OWNED & set(self.model_extra or {})
        if clashing:
            raise ValueError(
                f"loader keys {sorted(clashing)} are set by the framework "
                "(per-stage shuffle, the sample collate) and cannot be configured here."
            )
        return self


OptimizerConfig = ComponentConfig
"""The optimizer to build: a registry name ('adamw', 'sgd') or an import path, plus its arguments."""


class SchedulerConfig(ComponentConfig):
    """Which scheduler to build, and how Lightning should step it.

    Inherits the component grammar (``name`` / ``_target_`` / params); the
    policy fields below are declared, so they never leak into ``params`` and
    reach Lightning's ``lr_scheduler`` config instead of the constructor.
    """

    interval: Literal["epoch", "step"] = Field(
        "epoch", description="Whether the schedule steps per epoch or per batch."
    )
    frequency: int = Field(1, ge=1, description="Step the schedule once every N intervals.")
    monitor: str | None = Field(
        None,
        description="Logged metric a plateau schedule watches, e.g. 'val/loss' or 'val/label/accuracy'.",
    )
    strict: bool = Field(True, description="Fail when the monitored metric is absent instead of skipping the step.")


class TrainerConfig(BaseModel):
    """Trainer knobs; extras forward verbatim to ``lightning.Trainer``.

    ``profiler`` is the one knob that does not, because Lightning takes an object
    there and a section forwarding verbatim can only hand it a mapping. Declared,
    it is built the way every other component is — the same reason
    ``SchedulerConfig`` declares its policy fields.
    """

    model_config = ConfigDict(extra="allow")

    max_epochs: int = Field(10, gt=0, description="Epochs to train for.")
    accelerator: str = Field("auto", description="Device family ('cpu', 'gpu', 'auto').")
    devices: int | str = Field("auto", description="How many devices, or which ones.")
    profiler: ComponentConfig | None = Field(
        None,
        description=(
            "Where the run's wall clock went: 'simple', 'advanced' or 'pytorch', plus that "
            "profiler's own arguments. The report is written into the run's directory."
        ),
    )
