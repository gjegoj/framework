"""Data vocabulary of the framework: the plain containers that flow between ports."""

from __future__ import annotations

from collections.abc import Iterable, KeysView, Mapping
from dataclasses import dataclass, field
from functools import reduce
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, NotRequired, TypedDict

import torch

from src.core.log_keys import join

if TYPE_CHECKING:
    from torch import Tensor

    from src.core.ports import MetricSet
    from src.core.taxonomy import Objective, Stage, Topology


@dataclass(slots=True)
class Sample:
    """A single, un-batched example produced by the data layer.

    Values are intentionally loose (arrays, tensors, scalars): a sample exists
    before collation and may carry several input modalities and several task
    targets, each keyed by name.
    """

    CELLS: ClassVar[str] = "cells"
    """The metadata key the row's readable cells travel under.

    Named once so the writing side and ``Batch.cells`` cannot spell it differently.
    Input columns only — task names and input aliases are separate namespaces, so a
    target's source would need a key of its own rather than a silent collision here.
    """

    inputs: dict[str, Any]
    targets: dict[str, Any]
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

        ``meta`` stays a loose mapping because it genuinely is one: a datamodule
        wrapping a third-party collate passes that library's own keys through it. What
        is typed is the *reading* — the one key this framework writes gets one accessor
        that names it, types it and supplies its default.

        The shape is checked rather than assumed, because that same open seam lets a
        foreign collate put anything under this name, and an unexamined value would
        turn a key collision into a crash in whoever called ``len()`` on it first.
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
    """The objects a batch holds or predicted, concatenated across it.

    Flat rather than per-sample because that is the only shape a ragged quantity has
    that a tensor can carry: ``sample_index`` says which image each object belongs to.
    It is also the shape a detection collate already produces, so nothing is converted
    to satisfy this entity.

    Boxes are ``xyxy`` in pixels of the image as the model was fed it — one convention,
    pinned here, so a vendor's own dialect is converted inside that vendor's adapter and
    nowhere else.

    ``scores`` is ``None`` for ground truth, which has no confidence. That is what lets
    one entity serve both sides of a comparison, instead of two entities that would have
    to be kept in step by hand.
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

    Most of the framework works on tensors: a composed model's heads and criteria, the
    batch transforms that blend samples, an exported graph. A task predicting a set of
    objects reaches them only by being declared for a family they cannot serve, and the
    honest answer is to say which task and which reader rather than to fail deeper down
    on a missing attribute.
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

    Composite models put post-activation predictions into ``outputs``; a
    vendor family fills it with its native prediction structure. ``features``
    is kept for consumers that need the representation itself (metric
    learning, visualization); ``None`` when the producer does not expose it.

    ``logits`` are the same predictions before their activation, for the
    consumers that cannot work with what an activation leaves: a distillation
    temperature scales logits, and no rescaling recovers them from a
    distribution that already sums to one. ``None`` when the producer has no
    pre-activation form to show — a vendor model whose native output is boxes,
    for instance.
    """

    outputs: dict[str, TaskOutput]
    features: Features | None = None
    logits: dict[str, Tensor] | None = None


@dataclass(frozen=True, slots=True)
class Loss:
    """A loss value with its named components — single, weighted, or composite.

    One class covers every case: a criterion returns a single-part ``Loss``;
    weighting and multi-task aggregation are plain arithmetic, so no separate
    aggregator entity exists::

        total = Loss.sum(task.weight * loss.scoped(task.name) for ...)

    Instances are immutable; every operation returns a new ``Loss``. ``parts``
    keeps per-component values for logging while ``total`` is the scalar that
    is back-propagated.
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
    """What one model step yields: the loss to optimize, predictions and targets to judge.

    ``targets`` are metric-view targets by task name: the model owns target
    adaptation, so it hands metrics ready-to-compare values.
    """

    loss: Loss
    prediction: Prediction
    targets: dict[str, TaskOutput]


class LightningStepOutput(TypedDict):
    """What a training step hands back — Lightning's own contract, used as one.

    ``loss`` is what the loop backpropagates. ``preview`` rides along to every
    ``on_*_batch_end`` hook, because Lightning passes a step's return value there
    verbatim: a consumer that wants the batch's predictions is handed them by the
    framework, and nothing has to be kept, requested or invalidated.

    It is ``NotRequired`` because a preview is built only when a
    ``AwaitsPreview`` asked for this batch — holding one costs the activated
    outputs' storage through the optimizer step, and most steps of most runs have
    no reader. Absent means nobody asked; it does not mean the module cannot.
    """

    loss: Tensor
    preview: NotRequired[StepPreview]


@dataclass(frozen=True, slots=True)
class StepPreview:
    """What a step produced, detached — enough to draw it, and nothing that holds a graph.

    Not a ``StepResult``. Returning the result itself would carry the loss's
    ``grad_fn``, the outputs with ``requires_grad``, and every feature stream the
    backbone made — 352 MB of outputs alone for a ``[16, 21, 512, 512]``
    segmentation batch, measured.

    It carries no stage and no batch index. Whoever reads it is inside the hook
    for that stage and was handed that index; naming them here would be a second
    copy of facts the caller already has.
    """

    KEY: ClassVar[str] = "preview"
    """The key it rides under in a step's return value; the writer and the reader agree here."""

    outputs: dict[str, TaskOutput]
    targets: dict[str, TaskOutput]


def preview_of(step_output: object) -> StepPreview | None:
    """The preview a step returned, or ``None`` when the module returned something else.

    Lightning types a batch-end hook's ``outputs`` as ``Tensor | Mapping | None``,
    so the reading is typed here rather than at each call site — the same bargain
    ``Batch.cells`` makes for metadata.
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
        """The adapted target of a structure-supervised task (metric learning).

        Both views are empty and are never consumed by such a task's
        criterion or metrics — supervision comes from batch structure.
        """
        return cls(for_loss=torch.empty(0), for_metrics=torch.empty(0))


@dataclass(frozen=True, slots=True, eq=False)
class Task:
    """One learned objective, described in family-agnostic terms.

    A task is what an experiment learns and how it is judged: its two axes
    (``topology`` x ``objective``), its share of the total loss, and its
    metrics per stage. How predictions are produced is the model family's
    business — a composite model binds the task name to its per-task
    components; a vendor model binds it internally. ``batch.targets[task.name]``
    is the task's raw target.
    """

    name: str
    topology: Topology
    objective: Objective
    metrics: Mapping[Stage, MetricSet]
    weight: float = 1.0
    lr: float | None = None
    """Own learning rate for this task's bricks — its head and its criterion.

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

    Zero-count classes are kept: a class the training split never shows is the
    single most useful line in this table, and dropping it would make the column
    look healthy.

    ``counts`` sums to the number of rows for a single-label column and to more
    than that for a multilabel one, where a row carries several. For a mask it
    counts pixels, so the totals are large and the shares are what to read.
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

    A record rather than a bare nested dict. The row counts are here because the
    first question of any dataset report is *how much*, and because a split that
    went wrong — an empty stage, a test set larger than the train one — shows up
    there and nowhere else. Reading them off the target distributions instead
    would not work: a multilabel column counts more labels than it has rows.
    """

    rows: dict[Stage, int] = field(default_factory=dict)
    targets: dict[str, dict[Stage, Distribution]] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Whether there is anything at all to report."""
        return bool(self.rows or self.targets)


@dataclass(frozen=True, slots=True)
class TargetFacts:
    """What profiling the data revealed about one task's target.

    One task's frozen slice of the ``DataProfile`` — that is the whole
    difference between the two. The profile is the mutable box the data layer
    fills for every task; this is what a single objective is handed to build its
    bricks with, so it can neither see nor rewrite the facts of a task that is
    not its own.

    The bricks of an objective are built from these rather than from config:
    a head sizes itself from ``num_classes``, and ``class_values`` — the number
    each output position stands for — is what lets an ordered set of classes be
    read back as a single value. Absent facts are ``None``: a plain regression
    target has no classes, a categorical one has no values.
    """

    num_classes: int | None = None
    class_names: list[str] | None = None
    class_values: list[float] | None = None


@dataclass(slots=True)
class DataProfile:
    """Facts inferred from the data, filled at setup time and read at assembly time.

    The profile is the ordering contract that keeps runtime values out of
    config: the data layer writes facts while it fits encoders; tasks, heads,
    and criteria are built afterwards and read concrete values from here.

    It collects; it is not what consumers are handed. A builder takes one task's
    ``TargetFacts`` (see ``facts``) instead, which is frozen and covers that task
    alone. One record per task rather than a dict per fact: which facts a target
    happens to have is already modelled by ``TargetFacts`` (absent is ``None``),
    so a new kind of fact is declared there and reaches a profile untouched.
    """

    records: dict[str, TargetFacts] = field(default_factory=dict)

    def record(self, task_name: str, facts: TargetFacts) -> None:
        """Store what profiling one task's target revealed.

        One profiler owns one target, so a second record replaces rather than
        merges — the last profiling is the truth about that target.
        """
        self.records[task_name] = facts

    def facts(self, task_name: str) -> TargetFacts:
        """Everything profiling revealed about one task's target, as one value.

        An unprofiled task reads as facts without any: consumers that cannot
        work without one say so themselves, loudly and by name.
        """
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
