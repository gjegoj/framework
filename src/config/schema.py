"""Declaration validation only; runtime components receive values and ready objects."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
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


def refuse_a_path_that_is_not_there(key: str, path: str | None) -> None:
    """A declared file is answered for where it is declared, not minutes later where something opens it.

    One home because three declarations name one: the weights a run starts from, the weights a teacher
    answers with, and the vocabulary a task reads out of a file. Each would otherwise be found missing
    after the sources were read and the cache warmed — the vocabulary latest of all, since it is opened
    while the encoders are built.
    """
    if path is not None and not Path(path).is_file():
        raise ValueError(f"{key} names no file: {path}")


class ClassFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file: str = Field(min_length=1)

    @model_validator(mode="after")
    def present(self) -> ClassFile:
        refuse_a_path_that_is_not_there("classes.file", self.file)
        return self


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
    """A head and the feature streams it reads — one, or several where a task is learned over them together.

    ``stream`` rather than ``input``: a run's inputs are what the data feeds the model
    (``preprocessing.inputs.image``), while this names the features a backbone publishes.

    Several is how a pairing is declared — ``stream: [image_pooled, text_pooled]`` builds the declared
    head once per stream, at the width each of them publishes, and the order written is the order the
    answers arrive in. One or several is one key rather than two, as ``TaskConfig.loss`` already is.
    """

    stream: str | list[str] | None = None

    @property
    def streams(self) -> tuple[str, ...]:
        """The features this head reads, as whoever builds it sees them however the declaration spelled it."""
        if self.stream is None:
            return ()
        return (self.stream,) if isinstance(self.stream, str) else tuple(self.stream)

    @field_validator("stream")
    @classmethod
    def named_streams(cls, value: str | list[str] | None) -> str | list[str] | None:
        """Every spelling checked in one place, because there is one rule and two ways of writing it."""
        names = (value,) if isinstance(value, str) else tuple(value or ())
        if value is not None and not names:
            raise ValueError("A head reads at least one feature; a list naming none builds nothing.")
        if any(not name or name.strip() != name for name in names):
            raise ValueError("A head reads feature names, each of them unpadded.")
        if len(set(names)) != len(names):
            raise ValueError("A head reads distinct features: one named twice would build two heads over it.")
        return value


class TaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: ComponentConfig
    target_column: str | None = None
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


class TeacherConfig(ModelConfig):
    """A second network for a run to learn from, and the weights that make it worth learning from.

    An ordinary model declaration, because that is what it is: the same grammar, built by the same
    builder, sized by the same tasks. Its heads are not written here for exactly that reason — a teacher
    answers the questions this run asks, and a second statement of their shapes could disagree.

    The weights are not optional, and that is the whole of what this section adds. A network whose head
    was only just initialised answers with noise, and a run distilling from it would descend towards
    nothing while every number it reports looks ordinary. The file is one this framework wrote, since
    that is what a run's own checkpoint is; weights in someone else's shape are a phase of their own.
    """

    checkpoint_path: str = Field(min_length=1, description="The run whose weights this teacher answers with.")

    @model_validator(mode="after")
    def taught(self) -> TeacherConfig:
        refuse_a_path_that_is_not_there("learner.teacher.checkpoint_path", self.checkpoint_path)
        return self


class LearnerConfig(ComponentConfig):
    """The algorithm a run trains by, and the child positions an algorithm that needs them fills.

    Both belong to the algorithm rather than to the run, which is why they are written under it: a
    second network means nothing beside a learner that never asks it anything, and an objective
    measuring the distance to one means nothing without the network. Declared as sections of their own
    they were two halves held together by a refusal; declared here, the half cannot be written without
    naming the algorithm it belongs to.

    ``loss`` is not a task's. A task's is what its target is compared with, and every run has one per
    task; this is what an algorithm adds *beside* them — the distance to a second network, for the one
    that learns from one. Left out, an algorithm that reads an objective says what it defaults to, in
    the same place a task kind says it. ``teacher`` has no such default: a network to learn from cannot
    be derived from anything the run already holds.
    """

    loss: ComponentConfig | None = None
    teacher: TeacherConfig | None = None


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
