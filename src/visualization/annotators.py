"""Annotation: a task's step tensors become labels and a verdict."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Any, ClassVar, override

import numpy as np

from src.core.registry import named_by
from src.core.taxonomy import Objective, Topology
from src.visualization.entities import (
    Classification,
    Classifications,
    Regression,
    SampleView,
    Score,
    Segmentation,
    SegmentationClass,
    Verdict,
)
from src.visualization.registry import (
    annotation_objective_registry,
    annotation_topology_registry,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from torch import Tensor

    from src.core.entities import Task

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, eq=False)
class ClassPresence:
    """One class that holds, where it holds, and how strongly.

    ``where`` is boolean over the sample's positions — ``()``-shaped for a GLOBAL
    task (the class simply holds) and ``[H, W]`` for a DENSE one. That one array
    is what lets a topology, not an objective, decide between a chip and a mask.
    ``confidence`` is ``None`` on the ground-truth side, which expressed none.
    """

    index: int
    where: np.ndarray
    confidence: float | None = None


@dataclass(frozen=True, slots=True, eq=False)
class ClassReading:
    """Which classes hold; ``singular`` is what the objective allows, not what it found."""

    presences: tuple[ClassPresence, ...]
    singular: bool


@dataclass(frozen=True, slots=True, eq=False)
class ValueReading:
    """A number per position: ``()`` for a regressed scalar, ``[H, W]`` for a field."""

    values: np.ndarray


type Reading = ClassReading | ValueReading


class AnnotationObjective(ABC):
    """How one ``Objective`` reads predictions and targets, on any topology.

    ``reading`` says which kind it produces. Declared rather than inferred, because
    it is asked *before* a reading exists — at assembly, when a topology is asked
    whether it can label what this objective will read.

    ``scores`` is one sample's **activated** output, and whether it still has a
    class axis is the objective's own business — the activation decides:

    | objective | activation | one sample's shape |
    |---|---|---|
    | multiclass | ``softmax_probabilities`` | ``[C, *positions]`` |
    | multilabel | ``sigmoid_probabilities`` | ``[C, *positions]`` |
    | binary | ``sigmoid_probabilities`` | ``[*positions]`` — one channel, squeezed |
    | continuous | ``squeeze_single_output`` | ``[*positions]`` |

    Measured, not assumed: ``sigmoid_probabilities`` runs ``squeeze_single_output``
    first, so a binary head's ``[B, 1]`` arrives as ``[B]`` and its ``[B, 1, H, W]``
    as ``[B, H, W]``. A reader written against the logits shape indexes ``[0]``
    into that and gets an ``IndexError`` on a global task — or, worse, the first
    *row* of the map on a dense one, silently.

    ``target`` is the sample's metric-view target, which arrives hard: indices,
    multi-hot, or values.
    """

    reading: ClassVar[type[ClassReading | ValueReading]]

    @abstractmethod
    def read_output(self, scores: np.ndarray) -> Reading:
        """What the model said about this sample."""

    @abstractmethod
    def read_target(self, target: np.ndarray) -> Reading:
        """What is true of this sample."""


class AnnotationTopology(ABC):
    """How one ``Topology`` turns a pair of readings into labels and a verdict.

    One method per kind of reading, each defaulting to "this topology has no label
    for that". A topology overrides the ones it draws and says nothing about the
    rest — so a new reading kind adds one defaulted method here and touches no
    existing topology, where a ``match`` inside each of them had to grow a case in
    every one.

    ``draws`` is **derived** from those overrides rather than declared beside them.
    Declared, it could say yes where no branch existed, and the run found out an
    epoch in; derived, the two cannot disagree.
    """

    def label_classes(self, view: SampleView, task: Task, truth: ClassReading, predicted: ClassReading) -> None:
        """Draw what classes hold, where."""
        raise _no_label(self, ClassReading)

    def label_values(self, view: SampleView, task: Task, truth: ValueReading, predicted: ValueReading) -> None:
        """Draw a number, or a field of them."""
        raise _no_label(self, ValueReading)

    def draws(self, reading: type[ClassReading | ValueReading]) -> bool:
        """Whether this topology has a label for that kind of reading.

        Derived from which method the subclass overrode, so it cannot claim a
        pairing no branch exists for — the same guarantee the retired string
        table gave, without the table.
        """
        mine, base = type(self), AnnotationTopology
        if reading is ClassReading:
            return mine.label_classes is not base.label_classes
        return mine.label_values is not base.label_values

    def annotate(self, view: SampleView, task: Task, truth: Reading, predicted: Reading) -> None:
        """Route one sample's pair of readings to the labeller for their kind.

        The fallthrough covers both wrong pairs: two readings of different
        kinds, and a matched pair of a kind this router has no arm for — a new
        ``Reading`` member is refused here by name until it brings its labeller.
        """
        match truth, predicted:
            case ClassReading(), ClassReading():
                self.label_classes(view, task, truth, predicted)
            case ValueReading(), ValueReading():
                self.label_values(view, task, truth, predicted)
            case _:
                raise TypeError(
                    f"Task '{task.name}': ground truth read as {type(truth).__name__} and the prediction "
                    f"as {type(predicted).__name__}; one objective must produce both."
                )


def _no_label(topology: AnnotationTopology, reading: type[ClassReading | ValueReading]) -> TypeError:
    return TypeError(
        f"{type(topology).__name__} has no label for a {reading.__name__}. "
        f"Its draws() should have refused this pairing before the run started."
    )


def _presence(index: int, where: np.ndarray, scores: np.ndarray) -> ClassPresence:
    """Confidence is the mean score where the class holds: one value at a point, an average over a region."""
    return ClassPresence(index=index, where=where, confidence=float(scores[index][where].mean()))


@annotation_objective_registry.register(Objective.MULTICLASS)
class MulticlassAnnotation(AnnotationObjective):
    """Argmax over the class axis — outputs arrive activated, so no softmax here."""

    reading: ClassVar[type[ClassReading]] = ClassReading

    @override
    def read_output(self, scores: np.ndarray) -> Reading:
        winner = scores.argmax(axis=0)
        return ClassReading(
            presences=tuple(_presence(int(index), winner == index, scores) for index in np.unique(winner)),
            singular=True,
        )

    @override
    def read_target(self, target: np.ndarray) -> Reading:
        return ClassReading(
            presences=tuple(ClassPresence(int(index), target == index) for index in np.unique(target)),
            singular=True,
        )


@annotation_objective_registry.register(Objective.BINARY)
class BinaryAnnotation(AnnotationObjective):
    """Thresholds the one sigmoid value: argmax over a length-1 axis always answers class 0.

    Parameters:
        threshold (float): Above this the positive class holds.
    """

    reading: ClassVar[type[ClassReading]] = ClassReading

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold

    @override
    def read_output(self, scores: np.ndarray) -> Reading:
        # `scores` IS the positive probability — the activation squeezed the one
        # channel away — so there is no axis to index into here.
        positive = scores >= self._threshold
        # Both classes get a score, so a negative prediction reads as confident too.
        both = np.stack([1.0 - scores, scores])
        return ClassReading(presences=_sides(positive, both), singular=True)

    @override
    def read_target(self, target: np.ndarray) -> Reading:
        return ClassReading(presences=_sides(target > 0.5, None), singular=True)


def _sides(positive: np.ndarray, scores: np.ndarray | None) -> tuple[ClassPresence, ...]:
    """The negative and positive classes, each kept only where it holds anywhere."""
    return tuple(
        ClassPresence(index, where) if scores is None else _presence(index, where, scores)
        for index, where in ((0, ~positive), (1, positive))
        if where.any()
    )


@annotation_objective_registry.register(Objective.MULTILABEL)
class MultilabelAnnotation(AnnotationObjective):
    """Independent per-class probabilities: every channel above the threshold holds.

    Parameters:
        threshold (float): Above this a class is counted as predicted.
    """

    reading: ClassVar[type[ClassReading]] = ClassReading

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold

    @override
    def read_output(self, scores: np.ndarray) -> Reading:
        holds = scores >= self._threshold
        return ClassReading(
            presences=tuple(
                _presence(index, holds[index], scores) for index in range(holds.shape[0]) if holds[index].any()
            ),
            singular=False,
        )

    @override
    def read_target(self, target: np.ndarray) -> Reading:
        holds = target > 0.5
        return ClassReading(
            presences=tuple(
                ClassPresence(index, holds[index]) for index in range(holds.shape[0]) if holds[index].any()
            ),
            singular=False,
        )


@annotation_objective_registry.register(Objective.CONTINUOUS)
class ContinuousAnnotation(AnnotationObjective):
    """The activation already collapsed the class axis, so what arrives is the field itself."""

    reading: ClassVar[type[ValueReading]] = ValueReading

    @override
    def read_output(self, scores: np.ndarray) -> Reading:
        return ValueReading(values=scores)

    @override
    def read_target(self, target: np.ndarray) -> Reading:
        return ValueReading(values=target)


IOU = "iou"
"""What ``metric_registry`` calls intersection-over-union, and so what the page calls it.

The value stays ours, and is computed to agree. Measured against
``MulticlassJaccardIndex``: with ``average="macro"`` it returns exactly the mean
over the classes either side shows, and with ``ignore_index`` set it drops the void
pixels from every class's union — which is why ``_mean_iou`` masks both sides
rather than only skipping the void class. ``mae`` is ``abs(pred - gt)`` either way.

What is not borrowed is the metric *object*. Layering is the first reason: no
capability here imports another, and reaching into ``metrics/`` from ``visualization/``
would be the first. The second is configuration: the segmentation preset declares
``average="none"`` because the epoch report wants a per-class vector, so the task's
own objects would need a second configuration for the per-sample case — a second
source of truth for exactly what this shared naming removes.
"""

MAE = "mae"
"""What ``metric_registry`` calls the mean absolute error, and so what the page calls it."""


def _class_name(names: Sequence[str] | None, index: int) -> str:
    """``class{i}`` when a class has no declared name — the fallback ``Task.class_names`` names.

    ``core.reporting`` labels the same class the same way on the metric leaves and
    the confusion matrix, so a run does not end up with ``class3`` in the tracker's
    scalar list and a bare ``3`` on its sample grid.
    """
    return names[index] if names is not None and 0 <= index < len(names) else f"class{index}"


def _one(values: np.ndarray) -> float:
    """The single number a GLOBAL reading holds, whether it arrived 0-d or ``[1]``."""
    return float(values.reshape(-1)[0])


@annotation_topology_registry.register(Topology.GLOBAL)
class GlobalAnnotation(AnnotationTopology):
    """One prediction per sample: chips, judged by comparing what holds on each side."""

    @override
    def label_values(self, view: SampleView, task: Task, truth: ValueReading, predicted: ValueReading) -> None:
        true_value = _one(truth.values)
        predicted_value = _one(predicted.values)
        view.fields[(task.name, "gt")] = Regression(value=true_value)
        view.fields[(task.name, "pred")] = Regression(value=predicted_value)
        view.verdicts[task.name] = Verdict(scores=(Score(name=MAE, value=abs(predicted_value - true_value)),))

    @override
    def label_classes(self, view: SampleView, task: Task, truth: ClassReading, predicted: ClassReading) -> None:
        view.fields[(task.name, "gt")] = self._chips(truth, task.class_names)
        view.fields[(task.name, "pred")] = self._chips(predicted, task.class_names)
        # Set equality, so `correct` means *everything* matched — one class missing
        # or one extra is a miss, whatever the objective allows.
        correct = {entry.index for entry in truth.presences} == {entry.index for entry in predicted.presences}
        view.verdicts[task.name] = Verdict(correct=correct)

    @staticmethod
    def _chips(reading: ClassReading, names: Sequence[str] | None) -> Classification | Classifications:
        found = tuple(
            Classification(_class_name(names, entry.index), confidence=entry.confidence) for entry in reading.presences
        )
        # A single-label reading always holds exactly one class; an empty one draws
        # nothing rather than inventing a label for what the model did not say.
        return found[0] if reading.singular and found else Classifications(classifications=found)


@annotation_topology_registry.register(Topology.DENSE)
class DenseAnnotation(AnnotationTopology):
    """One prediction per location: masks, judged by mean IoU over the classes either side shows.

    It overrides ``label_classes`` and not ``label_values``: a field of numbers is a
    heatmap, and the IR has no label kind for one yet. Saying so is the whole of it —
    ``draws`` reads the overrides, so nothing has to be kept in step by hand.

    Parameters:
        ignore_index (int | None): A class drawn by neither side and judged by
            neither — the void label of a segmentation dataset.
    """

    def __init__(self, ignore_index: int | None = None) -> None:
        self._ignore_index = ignore_index

    @override
    def label_classes(self, view: SampleView, task: Task, truth: ClassReading, predicted: ClassReading) -> None:
        _refuse_mismatched_maps(task, truth, predicted)
        view.fields[(task.name, "gt")] = self._masks(truth, task.class_names)
        view.fields[(task.name, "pred")] = self._masks(predicted, task.class_names)
        overlap = self._mean_iou(truth, predicted)
        view.verdicts[task.name] = Verdict(scores=() if overlap is None else (Score(name=IOU, value=overlap),))

    def _masks(self, reading: ClassReading, names: Sequence[str] | None) -> Segmentation:
        return Segmentation(
            classes=tuple(
                SegmentationClass(_class_name(names, entry.index), entry.where)
                for entry in reading.presences
                if entry.index != self._ignore_index and entry.where.any()
            )
        )

    def _mean_iou(self, truth: ClassReading, predicted: ClassReading) -> float | None:
        """Averaged over the classes either side shows — a class absent from both says nothing.

        ``ignore_index`` has to drop the void *pixels*, not just the void class. A
        model cannot predict void, so every void pixel it labels lands in some real
        class's union and counts against it. Measured against
        ``MulticlassJaccardIndex(ignore_index=...)``: a prediction correct on every
        valid pixel scored 0.70 here while torchmetrics said 1.0, until the void was
        masked out of both sides. That agreement is the whole point of sharing the
        ``iou`` name with the epoch report.

        ``None`` when there is nothing left to measure — a sample that is all void
        earns no score at all, rather than a zero that would sort it as the worst on
        the page and drag the slider's floor down with it.
        """
        true_masks = {entry.index: entry.where for entry in truth.presences}
        predicted_masks = {entry.index: entry.where for entry in predicted.presences}
        shown = sorted((set(true_masks) | set(predicted_masks)) - {self._ignore_index})
        if not shown:
            return None
        void = true_masks.get(self._ignore_index) if self._ignore_index is not None else None
        empty = np.zeros(next(iter(chain(true_masks.values(), predicted_masks.values()))).shape, dtype=bool)
        overlaps = [_iou(true_masks.get(index, empty), predicted_masks.get(index, empty), void) for index in shown]
        measured = [value for value in overlaps if value is not None]
        return float(np.mean(measured)) if measured else None


def _refuse_mismatched_maps(task: Task, truth: ClassReading, predicted: ClassReading) -> None:
    """A head predicting at a resolution its label does not share is a bug, not a score.

    Left to numpy this is either a bare broadcast error from inside a batch-end hook,
    naming neither the task nor either shape, or — when one side happens to broadcast
    into the other — a silent IoU of 1.0 for a model that is not perfect.
    """
    shapes = {entry.where.shape for entry in (*truth.presences, *predicted.presences)}
    if len(shapes) > 1:
        raise ValueError(
            f"Task '{task.name}': its maps do not share a shape — got {sorted(shapes)}. "
            f"The head's output and its target must be at the same resolution to be compared."
        )


def _iou(left: np.ndarray, right: np.ndarray, void: np.ndarray | None = None) -> float | None:
    """``None`` where neither side claims a pixel — an empty union is not a score of zero."""
    if void is not None:
        left, right = left & ~void, right & ~void
    union = int((left | right).sum())
    return int((left & right).sum()) / union if union else None


class Annotator:
    """One task's annotation: its objective reads, its topology draws.

    A plain composer rather than an ABC — there is nothing to override. New
    behaviour lands as a new member in one of the two registries.
    """

    def __init__(self, objective: AnnotationObjective, topology: AnnotationTopology) -> None:
        self._objective = objective
        self._topology = topology

    def annotate(self, view: SampleView, task: Task, outputs: Tensor, targets: Tensor, index: int) -> None:
        """Label batch element ``index``; ``outputs`` are the task's activated outputs."""
        truth = self._objective.read_target(_numpy(targets[index]))
        predicted = self._objective.read_output(_numpy(outputs[index]))
        self._topology.annotate(view, task, truth, predicted)


def _numpy(tensor: Tensor) -> np.ndarray:
    array: np.ndarray = tensor.detach().cpu().float().numpy()
    return array


def build_annotators(tasks: Sequence[Task], **offered: Any) -> dict[str, Annotator]:
    """One annotator per drawable task; knobs reach the constructors that name them.

    Split on the same two axes the task model is: an ``AnnotationObjective`` reads
    predictions off the class axis and never looks at the trailing shape, so one
    implementation serves a GLOBAL ``[C]`` output and a DENSE ``[C, H, W]`` one alike,
    while an ``AnnotationTopology`` turns those readings into labels and a verdict — a
    point becomes a chip, a plane becomes masks. ``AnnotationTopology.draws`` mirrors
    ``TaskTopology.supports``. Keying one registry by the *pair* instead would duplicate
    along the axis it flattened — two argmaxes and two thresholds across four classes.

    Knobs (``threshold``, ``ignore_index``) are offered to every constructor and reach
    the ones that name them, the pattern derived values follow.

    A task that draws nothing is skipped with one line naming the task **and the
    reason** — an objective with no per-sample label to show, a topology with nothing
    per-sample at all, and a pairing the IR has no label for are three different
    situations, and a run must never silently draw none of them.
    """
    built: dict[str, Annotator] = {}
    for task in tasks:
        objective = _drawing_objective(task, offered)
        if objective is None:
            continue
        topology = _drawing_topology(task, objective, offered)
        if topology is None:
            continue
        built[task.name] = Annotator(objective=objective, topology=topology)
    return built


def _drawing_objective(task: Task, offered: Mapping[str, Any]) -> AnnotationObjective | None:
    if task.objective not in annotation_objective_registry:
        log.info(
            "Task '%s' is supervised by '%s', which has no per-sample label to show; "
            "it will not appear in the sample grid.",
            task.name,
            task.objective,
        )
        return None
    return _with_knobs(annotation_objective_registry.get(task.objective), offered)


def _drawing_topology(
    task: Task, objective: AnnotationObjective, offered: Mapping[str, Any]
) -> AnnotationTopology | None:
    if task.topology not in annotation_topology_registry:
        log.info(
            "Task '%s' has topology '%s', whose predictions are not per-sample; it will not appear in the sample grid.",
            task.name,
            task.topology,
        )
        return None
    topology = _with_knobs(annotation_topology_registry.get(task.topology), offered)
    if not topology.draws(objective.reading):
        log.info(
            "Task '%s': a '%s' task supervised by '%s' has no label kind in the visualization IR yet; "
            "it will not appear in the sample grid.",
            task.name,
            task.topology,
            task.objective,
        )
        return None
    return topology


def _with_knobs[T](factory: Callable[..., T], offered: Mapping[str, Any]) -> T:
    """Offer every knob, build with the ones this factory names."""
    return factory(**named_by(factory, offered))
