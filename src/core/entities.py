"""Data vocabulary of the framework: the plain containers that flow between ports."""

from __future__ import annotations

from collections.abc import Iterable, KeysView, Mapping
from dataclasses import dataclass, field
from functools import reduce
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, NotRequired, TypedDict

import torch

from src.core.log_keys import join

# Runtime, not TYPE_CHECKING: ``InputTopology.SINGLE`` is a dataclass field default,
# evaluated when this module loads; ``OutputTopology`` is imported beside it.
from src.core.taxonomy import InputTopology, OutputTopology

if TYPE_CHECKING:
    from torch import Tensor

    from src.core.ports import MetricSet
    from src.core.taxonomy import Objective, Stage


@dataclass(slots=True)
class Sample:
    """A single, un-batched example produced by the data layer.

    Values are loose (arrays, tensors, scalars): a sample exists before collation and may
    carry several inputs and several task targets, each keyed by name.
    """

    CELLS: ClassVar[str] = "cells"
    """The metadata key the row's readable cells travel under.

    Named once so the writing side and ``Batch.cells`` cannot spell it differently.
    Input columns only — task names and input aliases are separate namespaces, so a
    target's source would need a key of its own rather than a silent collision here.
    """

    inputs: dict[str, Any]
    targets: dict[str, Any]
    auxiliary_inputs: dict[str, Any] = field(default_factory=dict)
    """Arrays only the augmentations read — a mask that bounds a colour shift.

    Not model inputs and not targets: nothing is learned from them, and nothing
    downstream consumes them. ``collate_samples`` builds a ``Batch`` from ``inputs``,
    ``targets`` and ``meta`` alone, so whatever is stored here dies with the sample — no
    memory is spent moving it to a device, and forgetting to drop it is not a mistake
    anyone can make. A mask the model should *consume* is a regular input declared
    with the ``mask`` loader; one it should *learn from* is a task's target.
    """

    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Batch:
    """A collated batch of samples, ready for the model.

    ``targets`` are keyed by task name; ``meta`` carries per-sample provenance
    (e.g. source paths) that never enters the autograd graph.
    """

    inputs: dict[str, Tensor]
    targets: dict[str, TaskOutput]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def cells(self) -> list[dict[str, str]]:
        """Each sample's readable row cells, or an empty list where a source had none.

        ``meta`` stays a loose mapping (a third-party collate passes its own keys through it);
        the one key this framework writes gets a typed accessor. The shape is checked because a
        foreign collate could put anything under this name.
        """
        cells = self.meta.get(Sample.CELLS)
        if not isinstance(cells, list) or not all(isinstance(row, dict) for row in cells):
            return []
        rows: list[dict[str, str]] = cells
        return rows

    def to(self, device: torch.device | str) -> Batch:
        """Return a new ``Batch`` with all tensors moved to ``device``."""
        return Batch(
            inputs={name: tensor.to(device) for name, tensor in self.inputs.items()},
            targets={name: tensor.to(device) for name, tensor in self.targets.items()},
            meta=self.meta,
        )


@dataclass(slots=True)
class Features:
    """Named feature streams produced by a backbone.

    Simple backbones expose one stream (``Stream.FEATURES``); multi-stream
    backbones (encoder/decoder, multi-encoder) expose several. Standard
    stream names live in ``taxonomy.Stream``; custom names are plain strings.
    """

    streams: dict[str, Tensor]

    def __getitem__(self, stream: str) -> Tensor:
        try:
            return self.streams[stream]
        except KeyError:
            available = ", ".join(sorted(self.streams)) or "none"
            raise KeyError(f"Unknown feature stream '{stream}'. Available streams: {available}.") from None

    def __contains__(self, stream: str) -> bool:
        return stream in self.streams

    def keys(self) -> KeysView[str]:
        return self.streams.keys()


@dataclass(frozen=True, slots=True)
class Instances:
    """The objects a batch holds or predicted, flat across it.

    ``sample_index`` says which image each object belongs to. Boxes are ``xyxy`` in pixels of
    the image as the model was fed it — one convention, so a vendor's dialect is converted
    inside that vendor's adapter. ``scores`` is ``None`` for ground truth, which lets one
    entity serve both sides of a comparison.
    """

    boxes: Tensor
    labels: Tensor
    sample_index: Tensor
    scores: Tensor | None = None

    def __post_init__(self) -> None:
        counted = {"boxes": len(self.boxes), "labels": len(self.labels), "sample_index": len(self.sample_index)}
        if self.scores is not None:
            counted["scores"] = len(self.scores)
        if len(set(counted.values())) > 1:
            spelled = ", ".join(f"{count} {name}" for name, count in counted.items())
            raise ValueError(f"Instances columns must be the same length, got {spelled}.")

    def of(self, index: int) -> Instances:
        """The one image's objects, in the same entity — what a per-image consumer reads."""
        selected = self.sample_index == index
        return Instances(
            boxes=self.boxes[selected],
            labels=self.labels[selected],
            sample_index=self.sample_index[selected],
            scores=None if self.scores is None else self.scores[selected],
        )

    def detach(self) -> Instances:
        """Off the graph, the way a tensor output is — a preview detaches either alike."""
        return Instances(
            boxes=self.boxes.detach(),
            labels=self.labels.detach(),
            sample_index=self.sample_index.detach(),
            scores=None if self.scores is None else self.scores.detach(),
        )

    def to(self, device: torch.device | str) -> Instances:
        """Moved to a device, the way a tensor target is — ``Batch.to`` moves either alike."""
        return Instances(
            boxes=self.boxes.to(device),
            labels=self.labels.to(device),
            sample_index=self.sample_index.to(device),
            scores=None if self.scores is None else self.scores.to(device),
        )


def require_tensor(value: TaskOutput, *, task: str, wanted_by: str) -> Tensor:
    """A task's output where the reader can only serve a tensor, refused by name if not.

    Heads, criteria, batch transforms and an exported graph work on tensors; a task
    predicting a set of objects is refused naming the task and the reader.
    """
    if isinstance(value, Instances):
        raise TypeError(
            f"Task '{task}' predicts a set of objects, which {wanted_by} cannot serve. "
            f"A per-instance task belongs to a model family that owns its own head and loss."
        )
    return value


type TaskOutput = Tensor | Instances
"""What one task's prediction or target is: a tensor, or a set of objects per sample.

Named once rather than spelled out at each signature that carries it, so a third shape is
one edit here and a type error at every consumer that has not considered it — which is the
whole reason this is a union and not ``Any``.
"""


@dataclass(slots=True)
class Prediction:
    """Model output for one batch: per-task predictions in the family's shape.

    ``outputs`` are post-activation; ``logits`` the same before activation, for consumers an
    activation defeats (a distillation temperature scales logits). ``features`` is the
    representation itself. Either is ``None`` when the producer has no such form.
    """

    outputs: dict[str, TaskOutput]
    features: Features | None = None
    logits: dict[str, Tensor] | None = None


@dataclass(frozen=True, slots=True)
class Loss:
    """A loss value with its named components — single, weighted, or composite.

    Immutable; every operation returns a new ``Loss``. ``parts`` keeps per-component values
    for logging, ``total`` is the scalar that is back-propagated::

        total = Loss.sum(task.weight * loss.scoped(task.name) for ...)
    """

    total: Tensor
    parts: Mapping[str, Tensor]

    @classmethod
    def part(cls, name: str, value: Tensor) -> Loss:
        """Build a loss from one named part, e.g. ``Loss.part("ce", ce_value)``.

        The name is what the part appears under in logs
        (``train/{task}/ce``) and what keeps parts distinguishable when
        losses are added — criteria return named parts, never bare tensors.
        """
        return cls(total=value, parts={name: value})

    @classmethod
    def sum(cls, losses: Iterable[Loss]) -> Loss:
        """Fold losses into one; raises ``ValueError`` on an empty iterable."""
        materialized = list(losses)
        if not materialized:
            raise ValueError("Cannot sum an empty iterable of losses.")
        return reduce(lambda left, right: left + right, materialized)

    def scoped(self, scope: str) -> Loss:
        """Return the same loss with parts namespaced as ``"{scope}/{part}"``."""
        return Loss(
            total=self.total,
            parts={join(scope, name): value for name, value in self.parts.items()},
        )

    def __add__(self, other: Loss) -> Loss:
        collisions = self.parts.keys() & other.parts.keys()
        if collisions:
            names = ", ".join(sorted(collisions))
            raise ValueError(f"Loss parts collide on: {names}. Namespace one side with .scoped() first.")
        return Loss(total=self.total + other.total, parts={**self.parts, **other.parts})

    def __radd__(self, other: object) -> Loss:
        if other == 0:  # Lets built-in sum() start from its default 0.
            return self
        raise TypeError(f"Cannot add Loss to {type(other).__name__}.")

    def __mul__(self, weight: float) -> Loss:
        return Loss(
            total=self.total * weight,
            parts={name: value * weight for name, value in self.parts.items()},
        )

    __rmul__ = __mul__


class StepResult(NamedTuple):
    """What one model step yields: the loss to optimize, and predictions and targets for the metrics.

    ``targets`` are metric-view targets by task name: the model owns target adaptation, so it
    hands metrics ready-to-compare values.
    """

    loss: Loss
    prediction: Prediction
    targets: dict[str, TaskOutput]


class LightningStepOutput(TypedDict):
    """What a training step hands back — Lightning's own contract.

    ``loss`` is back-propagated. ``preview`` reaches every ``on_*_batch_end`` hook because
    Lightning passes the return value there verbatim; it is ``NotRequired`` because a preview
    is built only when an ``AwaitsPreview`` asked for this batch — holding one keeps the
    activated outputs alive through the optimizer step.
    """

    loss: Tensor
    preview: NotRequired[StepPreview]


@dataclass(frozen=True, slots=True)
class StepPreview:
    """What a step produced, detached — enough to draw it, nothing that holds a graph.

    Not the ``StepResult`` itself: that would carry the loss's ``grad_fn`` and every feature
    stream. Measured: 352 MB of outputs for a ``[16, 21, 512, 512]`` segmentation batch.
    """

    KEY: ClassVar[str] = "preview"
    """The key it is stored under in a step's return value; the writer and the reader agree here."""

    outputs: dict[str, TaskOutput]
    targets: dict[str, TaskOutput]


def preview_of(step_output: object) -> StepPreview | None:
    """The preview a step returned, or ``None`` when the module returned something else.

    Typed here rather than at each call site: Lightning types a hook's ``outputs`` as
    ``Tensor | Mapping | None``.
    """
    if isinstance(step_output, Mapping):
        preview = step_output.get(StepPreview.KEY)
        if isinstance(preview, StepPreview):
            return preview
    return None


@dataclass(frozen=True, slots=True)
class AdaptedTarget:
    """One raw target shaped into the two views a task consumes.

    The split exists because loss and metrics may need different encodings of
    the same target — e.g. MixUp trains against soft labels while metrics
    compare against hard class indices.
    """

    for_loss: Tensor
    for_metrics: Tensor

    @classmethod
    def absent(cls) -> AdaptedTarget:
        """The adapted target of a structure-supervised task (metric learning): both views empty."""
        return cls(for_loss=torch.empty(0), for_metrics=torch.empty(0))


@dataclass(frozen=True, slots=True, eq=False)
class Task:
    """One learned objective, described in family-agnostic terms.

    What an experiment learns and how it is evaluated: its axes (``output_topology`` x
    ``input_topology`` x ``objective``), its share of the total loss, and its metrics per
    stage. How predictions are produced is the model family's business.
    ``batch.targets[task.name]`` is the task's raw target.
    """

    name: str
    output_topology: OutputTopology
    objective: Objective
    metrics: Mapping[Stage, MetricSet]
    input_topology: InputTopology = InputTopology.SINGLE
    weight: float = 1.0
    lr: float | None = None
    """Own learning rate for this task's components — its head and its criterion.

    ``None`` shares the run's rate. Like ``weight``, a training knob is part of
    what a task *is*: how strongly it pulls, and how fast its own parts move.
    """

    class_names: list[str] | None = None
    """Names aligned with class indices, for per-class log leaves and matrix labels.

    ``None`` for class-free tasks, or when neither declaration nor fitting
    produced names; consumers fall back to ``class{i}``.
    """

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Task name must be non-empty.")
        if self.weight <= 0:
            raise ValueError(f"Task weight must be positive, got {self.weight}.")
        if self.lr is not None and self.lr <= 0:
            raise ValueError(f"Task lr must be positive, got {self.lr}.")


@dataclass(frozen=True, slots=True)
class ClassDistribution:
    """How many of each class a column holds — the imbalance, before it surprises anyone.

    Zero-count classes are kept: a class the training split never shows is the most useful
    line. ``counts`` sums to the row count for a single-label column, to more for a
    multilabel one, and to pixels for a mask.
    """

    counts: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def shares(self) -> dict[str, float]:
        """Each class as a fraction of the total; all zero when there is nothing to divide."""
        total = self.total
        return {name: (count / total if total else 0.0) for name, count in self.counts.items()}


@dataclass(frozen=True, slots=True)
class ValueDistribution:
    """The five-number summary of a numeric column, plus its mean and deviation.

    Quantiles rather than a histogram: the shape of a target is read from where its
    mass sits, and the quartiles say that in five numbers that fit a terminal row —
    where a histogram would need a bin count nobody has a principled value for.
    """

    count: int
    mean: float
    deviation: float
    minimum: float
    q25: float
    median: float
    q75: float
    maximum: float


type Distribution = ClassDistribution | ValueDistribution
"""What one target column looks like, in whichever of the two shapes fits it."""


@dataclass(frozen=True, slots=True)
class DatasetStatistics:
    """What a run is about to train on: how much of it there is, and what it holds.

    Row counts are here because a split that went wrong — an empty stage, a test set larger
    than train — shows up there and nowhere else.
    """

    rows: dict[Stage, int] = field(default_factory=dict)
    targets: dict[str, dict[Stage, Distribution]] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Whether there is anything at all to report."""
        return bool(self.rows or self.targets)


@dataclass(frozen=True, slots=True)
class TargetFacts:
    """What profiling the data revealed about one task's target.

    One task's frozen slice of the ``DataProfile``: an objective builds its components from
    these rather than from config — a head sizes itself from ``num_classes``, and
    ``class_values`` is what lets an ordered set of classes be read back as one value. Absent
    facts are ``None``.
    """

    num_classes: int | None = None
    class_names: list[str] | None = None
    class_values: list[float] | None = None


@dataclass(slots=True)
class DataProfile:
    """Facts inferred from the data, filled at setup time and read at assembly time.

    The ordering contract that keeps runtime values out of config: the data layer writes facts
    while it fits encoders; tasks, heads and criteria are built afterwards. Consumers are
    handed one task's frozen ``TargetFacts`` (see ``facts``), never the profile itself.
    """

    records: dict[str, TargetFacts] = field(default_factory=dict)

    def record(self, task_name: str, facts: TargetFacts) -> None:
        """Store what profiling one task's target revealed; a second record replaces the first."""
        self.records[task_name] = facts

    def facts(self, task_name: str) -> TargetFacts:
        """Everything profiling revealed about one task's target; an unprofiled task reads as facts without any."""
        return self.records.get(task_name, TargetFacts())

    def require_num_classes(self, task_name: str) -> int:
        """Return the class count for ``task_name`` or fail with the ordering hint."""
        num_classes = self.facts(task_name).num_classes
        if num_classes is None:
            raise LookupError(
                f"num_classes for task '{task_name}' is not profiled yet. "
                "Profile the data (setup) before assembling tasks and heads."
            )
        return num_classes
