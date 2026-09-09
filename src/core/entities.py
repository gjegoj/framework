"""Data vocabulary of the framework: the plain containers that flow between ports."""

from __future__ import annotations

from collections.abc import Iterable, KeysView, Mapping
from dataclasses import dataclass, field
from functools import reduce
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import torch

from src.core.log_keys import join

if TYPE_CHECKING:
    from torch import Tensor


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
    the image as the model was fed it — one convention, so a library's dialect is converted
    inside that library's adapter. ``scores`` is ``None`` for ground truth, which lets one
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


@dataclass(frozen=True, slots=True)
class TaskFacts:
    """What the data revealed about one task's target, once the pipeline was set up.

    One task's frozen slice of the ``DatasetFacts``: a kind builds its components from these
    rather than from config — a head sizes itself from ``num_classes``, ``class_names`` label
    the per-class leaves and the drawn classes, and ``class_values`` is what lets an ordered
    set of classes be read back as one value. Absent facts are ``None``.
    """

    num_classes: int | None = None
    class_names: tuple[str, ...] | None = None
    class_values: tuple[float, ...] | None = None


type DatasetFacts = Mapping[str, TaskFacts]
"""What ``setup`` learned about every task's target, by task name — returned, never filled behind a caller's back.

A task that reads no column (metric learning) has no entry; the composition root reads it
as facts without any. The ordering that keeps sizes out of config is then visible in one
line: ``facts = pipeline.setup()`` before any head is built.
"""
