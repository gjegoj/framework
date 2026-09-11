"""Declaration validation only; runtime components receive values and ready objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.core import validate_classes, validate_name


class ComponentConfig(BaseModel):
    """One grammar for naming something to build: ``name`` or ``_target_``, every other key a constructor argument.

    ``loss: cross_entropy`` reads as ``{name: cross_entropy}``. A nested mapping is an ordinary
    argument unless it carries ``_target_`` of its own; a nested position has no registry.

    ``name`` is the selector and therefore reserved: a constructor with a parameter of that name keeps
    its default, whichever way it is reached. It is the one argument a declaration cannot pass.
    """

    TARGET_KEY: ClassVar[str] = "_target_"

    model_config = ConfigDict(extra="allow")
    name: str | None = Field(None, min_length=1)
    import_path: str | None = Field(None, alias=TARGET_KEY, min_length=1)

    @model_validator(mode="before")
    @classmethod
    def shorthand(cls, value: Any) -> Any:
        return {"name": value} if isinstance(value, str) else value

    @model_validator(mode="after")
    def one_selector(self) -> ComponentConfig:
        if (self.name is None) == (self.import_path is None):
            raise ValueError(f"Declare exactly one of name or {self.TARGET_KEY}.")
        reserved = sorted(key for key in self.params if key.startswith("_") or key == "import_path")
        if reserved:
            raise ValueError(
                f"Unsupported keys {', '.join(reserved)}: {self.TARGET_KEY} names what to build and every "
                "other key is an argument for it, so there is nothing an underscored key could mean here."
            )
        return self

    @property
    def params(self) -> dict[str, Any]:
        """Everything but the selector: the constructor's keyword arguments, nested values untouched."""
        return dict(self.model_extra or {})

    @property
    def spelled(self) -> str:
        """The component as the declaration wrote it, for a message that names it."""
        return self.name if self.name is not None else str(self.import_path)


class ClassFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file: str = Field(min_length=1)


class WeightedLossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    loss: ComponentConfig
    weight: float = Field(1.0, ge=0, allow_inf_nan=False)
    log_name: str | None = Field(None, min_length=1)

    @field_validator("log_name")
    @classmethod
    def reportable(cls, value: str | None) -> str | None:
        """A term reports under this name, so it has to be one a metric key can carry."""
        if value is not None:
            validate_name(value, label="Loss log")
        return value


class HeadConfig(ComponentConfig):
    """A head and the one feature stream it reads; a head over several streams arrives with detection.

    ``stream`` rather than ``input``: a run's inputs are what the data feeds the model
    (``preprocessing.inputs.image``), while this names one of the features a backbone publishes.
    """

    stream: str | None = Field(None, min_length=1)

    @field_validator("stream")
    @classmethod
    def named_stream(cls, value: str | None) -> str | None:
        if value is not None and value.strip() != value:
            raise ValueError("A head reads one feature name, unpadded.")
        return value


class TaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: ComponentConfig
    target: str | None = None
    classes: dict[int, str] | ClassFile | None = None
    head: HeadConfig | None = None
    loss: ComponentConfig | list[WeightedLossConfig] | None = None
    target_encoder: ComponentConfig | None = None
    metrics: dict[str, ComponentConfig] | None = None
    weight: float = Field(1.0, gt=0, allow_inf_nan=False)
    lr: float | None = Field(None, gt=0, allow_inf_nan=False)

    @field_validator("metrics")
    @classmethod
    def reportable(cls, value: dict[str, ComponentConfig] | None) -> dict[str, ComponentConfig] | None:
        """The label is what a report shows a metric under, so it has to be one a metric key can carry."""
        for label in value or {}:
            validate_name(label, label="Metric")
        return value

    @field_validator("classes", mode="before")
    @classmethod
    def explicit_classes(cls, value: Any) -> Any:
        if value is None or isinstance(value, ClassFile):
            return value
        if not isinstance(value, Mapping):
            raise ValueError(  # noqa: TRY004 -- Pydantic wraps ValueError, not TypeError.
                "Classes require an index-to-name mapping or a file declaration."
            )
        if "file" in value:
            return ClassFile.model_validate(value)
        normalized: dict[int, str] = {}
        for key, name in value.items():
            if type(key) is int:
                index = key
            elif isinstance(key, str) and key.isascii() and key.isdecimal():
                index = int(key)
            else:
                raise ValueError(f"Invalid class index: {key!r}.")
            if index in normalized:
                raise ValueError(f"Colliding class index: {key!r}.")
            normalized[index] = name
        validate_classes(normalized)
        return normalized

    @model_validator(mode="after")
    def task_connections(self) -> TaskConfig:
        validate_losses(self.loss)
        return self


def validate_losses(loss: ComponentConfig | list[WeightedLossConfig] | None) -> None:
    if isinstance(loss, list):
        if not loss or not any(item.weight > 0 for item in loss):
            raise ValueError("A loss list needs at least one positive weight.")
        explicit = [item.log_name for item in loss if item.log_name is not None]
        if len(set(explicit)) != len(explicit):
            raise ValueError("Explicit loss log names must be distinct.")


class ModelConfig(ComponentConfig):
    """A network by name or import path; ``backbone`` is the child position a composite fills from its own registry."""

    backbone: ComponentConfig | None = None


class PreprocessingConfig(ComponentConfig):
    """Modality-specific loading, normalization and collation.

    ``inputs``, ``auxiliary_inputs``, ``collator`` and ``cache`` are child positions: each is a
    declaration in its own right, resolved against the registry that holds names for it.
    """

    inputs: dict[str, ComponentConfig] | None = None
    auxiliary_inputs: dict[str, ComponentConfig] | None = None
    collator: ComponentConfig | None = None
    cache: ComponentConfig | None = None

    @field_validator("inputs", "auxiliary_inputs")
    @classmethod
    def named_inputs(cls, value: dict[str, ComponentConfig] | None) -> dict[str, ComponentConfig] | None:
        for name in value or {}:
            validate_name(name, label="Input")
        return value
