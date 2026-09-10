"""Model adaptation and teacher composition reuse the ordinary component and loss grammar."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config.schema import AdaptersConfig, ComponentConfig, HeadConfig, WeightedLossConfig, validate_losses
from src.core.entities import validate_name


class TeacherConfig(BaseModel):
    """A frozen complete model, including its trained heads; never infer trained heads from a backbone.

    Input names refer to experiment preprocessing views. Different resolutions or
    tokenizers use different named views of the same source, not renormalized student tensors.
    """

    model_config = ConfigDict(extra="forbid")
    model: ComponentConfig
    heads: dict[str, HeadConfig] = Field(default_factory=dict)
    adapters: AdaptersConfig = Field(default_factory=list)
    checkpoint_path: str | None = Field(None, min_length=1)

    @model_validator(mode="after")
    def named_heads(self) -> TeacherConfig:
        for name in self.heads:
            validate_name(name, kind="Teacher head")
        if "heads" in self.model.params:
            raise ValueError("Declare teacher heads once, at teacher.heads.")
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
