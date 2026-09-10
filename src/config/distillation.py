"""Model adaptation and teacher composition reuse the ordinary component and loss grammar."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config.schema import (
    AdaptersConfig,
    ComponentConfig,
    ModelConfig,
    WeightedLossConfig,
    validate_losses,
)
from src.core import validate_name


class TeacherConfig(BaseModel):
    """A frozen model beside the student: this run's tasks on another network, or a model that arrives whole.

    A ``composite`` teacher gets one head per task from the same ``tasks`` declarations and
    facts as the student, so their outputs match by construction and no size is stated twice;
    a ``_target_`` teacher brings its own heads. Input names refer to preprocessing views, so a
    teacher may read a differently sized view of the same source.
    """

    model_config = ConfigDict(extra="forbid")
    model: ModelConfig
    adapters: AdaptersConfig = Field(default_factory=list)
    checkpoint_path: str | None = Field(None, min_length=1)

    @model_validator(mode="after")
    def heads_come_from_the_tasks(self) -> TeacherConfig:
        if "heads" in self.model.params:
            raise ValueError("A teacher's heads follow the run's tasks; declare none on the teacher.")
        return self


class DistillationConfig(BaseModel):
    """Decorate the selected strategy during training; the student alone serves inference.

    Each loss component chooses student outputs/features, a named teacher and its
    outputs/features. Aggregation and feature projections belong to that component;
    there is no implicit averaging or assumption of identical class vocabularies.
    """

    model_config = ConfigDict(extra="forbid")
    teachers: dict[str, TeacherConfig] = Field(min_length=1)
    loss: ComponentConfig | list[WeightedLossConfig]

    @model_validator(mode="after")
    def comparisons(self) -> DistillationConfig:
        for name in self.teachers:
            validate_name(name, kind="Teacher")
        validate_losses(self.loss)
        return self
