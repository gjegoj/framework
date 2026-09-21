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


def validate_streams(value: str | list[str] | None) -> str | list[str] | None:
    """Every spelling checked in one place, because there is one rule and two ways of writing it.

    Read by the two declarations that name a backbone's features: a head, which reads one or several,
    and a term of ``learner.loss``, which compares them between two networks. One home rather than a
    copy each — it is one rule about one kind of name, and the day it grows a case two copies would be
    free to grow it differently.
    """
    names = streams_named(value)
    if value is not None and not names:
        raise ValueError("A stream is named in order to be read; a list naming none names nothing.")
    if any(not name or name.strip() != name for name in names):
        raise ValueError("A stream is named by an unpadded name.")
    if len(set(names)) != len(names):
        raise ValueError("Streams are named distinctly: one written twice would be read twice.")
    return value


def streams_named(value: str | list[str] | None) -> tuple[str, ...]:
    """The features a declaration names, as whoever reads them sees them however they were spelled."""
    if value is None:
        return ()
    return (value,) if isinstance(value, str) else tuple(value)


class NamesStreams(BaseModel):
    """A declaration naming features a backbone publishes: one, or several where there is a real choice.

    Two positions write this word — a head, which reads what it names, and a term of ``learner.loss``,
    which compares it between two networks — and it means the same thing in both, so it is declared
    once. What each of them does with the names is what their own docstrings say.

    Measured on pydantic 2.13.4, mixing this in leaves each class's own ``extra`` policy standing:
    ``HeadConfig`` keeps ``allow`` and carries a head's arguments through, and a term keeps ``forbid``
    and refuses a misspelled key.
    """

    stream: str | list[str] | None = None

    @property
    def streams(self) -> tuple[str, ...]:
        """The features named, as whoever builds from this sees them however the declaration spelled them."""
        return streams_named(self.stream)

    @field_validator("stream")
    @classmethod
    def named_streams(cls, value: str | list[str] | None) -> str | list[str] | None:
        return validate_streams(value)


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


class DistilledLossConfig(WeightedLossConfig, NamesStreams):
    """One term of what a run learns from a second network, and which of its answers the term compares.

    ``stream`` is the word a head already writes, and it means here exactly what it means there: a name
    the backbone publishes. Named, this term compares that feature of the two networks — one term per
    name; left out, it compares their answers, which is what distilling here has always meant.

    ``weight`` is the share within what is learned from the teacher, and ``learner.weight`` is what that
    whole half is worth beside the tasks' own objectives — the two levels a task and its loss list
    already have. ``log_name`` is how a run tells two terms over one reading apart; left out, a term is
    named for what it reads rather than for the loss it uses, so that a column survives a change of
    measure.
    """


class HeadConfig(ComponentConfig, NamesStreams):
    """A head and the feature streams it reads — one, or several where a task is learned over them together.

    ``stream`` rather than ``input``: a run's inputs are what the data feeds the model
    (``preprocessing.inputs.image``), while this names the features a backbone publishes.

    Several is how a pairing is declared — ``stream: [image_pooled, text_pooled]`` builds the declared
    head once per stream, at the width each of them publishes, and the order written is the order the
    answers arrive in. One or several is one key rather than two, as ``TaskConfig.loss`` already is.

    ``checkpoint_path`` names weights written for this head and no other — the tail of a larger head a
    run continues, prepared as a file of its own. Typed here rather than left among the head's own
    arguments for the reason every child position is: an argument reaches the constructor, and no head
    should have to accept a knob about where its numbers came from.
    """

    checkpoint_path: str | None = Field(
        None, min_length=1, description="Weights for exactly this head, prepared wherever they came from."
    )

    @model_validator(mode="after")
    def prepared(self) -> HeadConfig:
        refuse_a_path_that_is_not_there("head.checkpoint_path", self.checkpoint_path)
        return self


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
    """A network by name or import path; ``backbone`` and ``neck`` are the child positions a composite
    fills from their own registries.

    ``neck`` is what a run puts between the two: a backbone reads a sample, a neck reads the features
    it published, and the heads are sized from whichever of them published last. Left out — which is
    every ordinary run — the heads read the backbone and nothing about the run changes, not even the
    checkpoint it writes.

    A position rather than a backbone wrapped around a backbone, because the paths a composite
    registers are a contract that `freeze`, `adapter` and every checkpoint address: wrapped, a
    projection moved every path under `backbone` one level down, and `modules: [backbone]` came to
    mean "the encoder and the projection" in a run that had written it to mean the encoder.
    """

    backbone: ComponentConfig | None = None
    neck: ComponentConfig | None = None


class TeacherConfig(ModelConfig):
    """A second network for a run to learn from, and the weights that make it worth learning from.

    An ordinary model declaration, because that is what it is: the same grammar, built by the same
    builder, sized by the same tasks.

    ``heads`` is the one thing a teacher may say about its own shape, and the widths stay derived
    either way — that is what the rule against declaring anything twice was protecting. How many
    numbers a head answers with is the task's, and how wide the features it reads are is its own
    backbone's; neither is written here. *Which* head reaches those numbers is a different question,
    and a run continuing the tail of a teacher's head has to be able to answer it: the whole of that
    head on the teacher, the tail alone on the student. Left out, the teacher answers through the
    heads this run's own model does.

    The weights are not optional, and that is the whole of what this section adds. A network whose head
    was only just initialised answers with noise, and a run distilling from it would descend towards
    nothing while every number it reports looks ordinary. The file is one this framework wrote, since
    that is what a run's own checkpoint is; weights in someone else's shape are a phase of their own.
    """

    checkpoint_path: str = Field(min_length=1, description="The run whose weights this teacher answers with.")
    heads: dict[str, HeadConfig] | None = None

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

    One term or several, written the way ``tasks.<name>.loss`` is written, because it is the same
    question asked of a second network. A term that names a ``stream`` compares that feature of the two
    networks, one term per name; a term that names none compares their answers. ``weight`` here is what
    the whole of what is learned from the teacher is worth beside the tasks' own objectives, and the
    share inside a term is its own — the two levels a task and its loss list already have.
    """

    loss: ComponentConfig | list[DistilledLossConfig] | None = None
    teacher: TeacherConfig | None = None

    @field_validator("loss")
    @classmethod
    def compared(
        cls, value: ComponentConfig | list[DistilledLossConfig] | None
    ) -> ComponentConfig | list[DistilledLossConfig] | None:
        """A single objective is that loss itself, so `stream` beside it would be its constructor's.

        Refused rather than read, because both spellings otherwise build: the loss takes the word as an
        argument it never declared, or swallows it in ``**kwargs``, and the run compares answers while
        the declaration says features. Only this word — ``weight`` beside a single objective is a real
        argument of real losses (``cross_entropy`` weights its classes by it), so what looks like the
        same slip there is a declaration this cannot tell from the genuine one.
        """
        if isinstance(value, ComponentConfig) and "stream" in value.params:
            raise ValueError(
                "`learner.loss` names one objective here, and every key beside it is that loss's own "
                "argument, so `stream` would reach its constructor. A term comparing a feature of the "
                f"two networks is an item of the list: `loss: [{{loss: {value.spelled}, stream: "
                f"{value.params['stream']!r}}}]`."
            )
        return value


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
