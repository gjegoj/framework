"""Annotation: a task's step tensors become labels and a verdict."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, override

import numpy as np

from src.visualization.entities import (
    Classification,
    Classifications,
    Regression,
    SampleView,
    Score,
    Segmentation,
    SegmentationClass,
    TaskView,
    Verdict,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor


@dataclass(frozen=True, slots=True)
class DrawingKnobs:
    """What a page lets the user tune about how outputs are read.

    One value handed to every kind: a kind reads the knob it needs and ignores the rest,
    so a new knob is one field here rather than a change to every kind's signature.

    Attributes:
        threshold: Above this a binary or multilabel class holds.
        ignore_index: A class a dense reading neither draws nor scores.
    """

    threshold: float = 0.5
    ignore_index: int | None = None


@dataclass(frozen=True, slots=True, eq=False)
class ClassPresence:
    """One class that holds, where it holds, and how strongly.

    ``where`` is boolean over the sample's positions — ``()``-shaped for a global
    output (the class simply holds) and ``[H, W]`` for a dense one. That one array
    is what lets the drawer, not the reader, decide between a chip and a mask.
    ``confidence`` is ``None`` on the ground-truth side, which expressed none.
    """

    index: int
    where: np.ndarray
    confidence: float | None = None


@dataclass(frozen=True, slots=True, eq=False)
class ClassReading:
    """Which classes hold; ``singular`` is what the kind allows, not what it found."""

    presences: tuple[ClassPresence, ...]
    singular: bool


@dataclass(frozen=True, slots=True, eq=False)
class ValueReading:
    """A number per position: ``()`` for a regressed scalar, ``[H, W]`` for a field."""

    values: np.ndarray


type Reading = ClassReading | ValueReading


class Reader(ABC):
    """How a kind's label semantics reads predictions and targets, on any shape of output.

    ``scores`` is one sample's **activated** output, and the activation decides whether it
    still has a class axis: multiclass and multilabel are ``[C, *positions]``; binary and
    continuous are ``[*positions]`` — measured, ``sigmoid_probabilities`` squeezes the single
    channel, so a reader indexing ``[0]`` into it would take the first row of a dense map.
    ``target`` arrives hard: indices, multi-hot, or values.
    """

    @abstractmethod
    def read_output(self, scores: np.ndarray) -> Reading:
        """What the model said about this sample."""

    @abstractmethod
    def read_target(self, target: np.ndarray) -> Reading:
        """What is true of this sample."""


class Drawer(ABC):
    """How one shape of output turns a pair of readings into labels and a verdict.

    One method per kind of reading, each defaulting to "no label for that"; a drawer
    overrides the ones it draws, and a kind composes a reader with a drawer that draws
    what the reader produces.
    """

    def label_classes(self, view: SampleView, task: TaskView, truth: ClassReading, predicted: ClassReading) -> None:
        """Draw what classes hold, where."""
        raise _no_label(self, ClassReading)

    def label_values(self, view: SampleView, task: TaskView, truth: ValueReading, predicted: ValueReading) -> None:
        """Draw a number, or a field of them."""
        raise _no_label(self, ValueReading)

    def annotate(self, view: SampleView, task: TaskView, truth: Reading, predicted: Reading) -> None:
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
                    f"as {type(predicted).__name__}; one reader must produce both."
                )


def _no_label(drawer: Drawer, reading: type[ClassReading | ValueReading]) -> TypeError:
    return TypeError(
        f"{type(drawer).__name__} has no label for a {reading.__name__}. "
        f"The kind that composed it should have paired it with a reader it can draw."
    )


def _presence(index: int, where: np.ndarray, scores: np.ndarray) -> ClassPresence:
    """Confidence is the mean score where the class holds: one value at a point, an average over a region."""
    return ClassPresence(index=index, where=where, confidence=float(scores[index][where].mean()))


class MulticlassReader(Reader):
    """Argmax over the class axis — outputs arrive activated, so no softmax here."""

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


class BinaryReader(Reader):
    """Thresholds the one sigmoid value: argmax over a length-1 axis always answers class 0.

    Parameters:
        threshold (float): Above this the positive class holds.
    """

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


class MultilabelReader(Reader):
    """Independent per-class probabilities: every channel above the threshold holds.

    Parameters:
        threshold (float): Above this a class is counted as predicted.
    """

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


class ValueReader(Reader):
    """The activation already collapsed the class axis, so what arrives is the field itself."""

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

What is not borrowed is the metric *object*. Layering is the first reason: this package
imports no other capability, and reaching into ``metrics/`` would be the first. The
second is configuration: the segmentation kind declares ``average="none"`` because the
epoch report wants a per-class vector, so the task's own objects would need a second
configuration for the per-sample case — a second source of truth for exactly what this
shared naming removes.
"""

MAE = "mae"
"""What ``metric_registry`` calls the mean absolute error, and so what the page calls it."""


def _class_name(names: Sequence[str] | None, index: int) -> str:
    """``class{i}`` when a class has no declared name — the fallback the task's facts leave.

    ``loggers.report`` labels the same class the same way on the metric leaves and
    the confusion matrix, so a run does not end up with ``class3`` in the tracker's
    scalar list and a bare ``3`` on its sample grid.
    """
    return names[index] if names is not None and 0 <= index < len(names) else f"class{index}"


def _one(values: np.ndarray) -> float:
    """The single number a global reading holds, whether it arrived 0-d or ``[1]``."""
    return float(values.reshape(-1)[0])


class GlobalDrawer(Drawer):
    """One prediction per sample: chips, matched by comparing what holds on each side."""

    @override
    def label_values(self, view: SampleView, task: TaskView, truth: ValueReading, predicted: ValueReading) -> None:
        true_value = _one(truth.values)
        predicted_value = _one(predicted.values)
        view.fields[(task.name, "gt")] = Regression(value=true_value)
        view.fields[(task.name, "pred")] = Regression(value=predicted_value)
        view.verdicts[task.name] = Verdict(scores=(Score(name=MAE, value=abs(predicted_value - true_value)),))

    @override
    def label_classes(self, view: SampleView, task: TaskView, truth: ClassReading, predicted: ClassReading) -> None:
        names = task.class_names
        view.fields[(task.name, "gt")] = self._chips(truth, names)
        view.fields[(task.name, "pred")] = self._chips(predicted, names)
        # Set equality, so `correct` means *everything* matched — one class missing
        # or one extra is a miss, whatever the kind allows.
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


class DenseDrawer(Drawer):
    """One prediction per location: masks, scored by mean IoU over the classes either side shows.

    It overrides ``label_classes`` and not ``label_values``: a field of numbers is a
    heatmap, and the IR has no label kind for one yet.

    Parameters:
        ignore_index (int | None): A class drawn by neither side and scored by neither.
    """

    def __init__(self, ignore_index: int | None = None) -> None:
        self._ignore_index = ignore_index

    @override
    def label_classes(self, view: SampleView, task: TaskView, truth: ClassReading, predicted: ClassReading) -> None:
        _refuse_mismatched_maps(task, truth, predicted)
        names = task.class_names
        view.fields[(task.name, "gt")] = self._masks(truth, names)
        view.fields[(task.name, "pred")] = self._masks(predicted, names)
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

        ``ignore_index`` drops the void *pixels*, not just the void class: measured against
        ``MulticlassJaccardIndex(ignore_index=...)``, a prediction correct on every valid pixel
        scored 0.70 here until the void was masked out of both sides. ``None`` when nothing is
        left to measure, rather than a zero that would sort the sample as the worst.
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


def _refuse_mismatched_maps(task: TaskView, truth: ClassReading, predicted: ClassReading) -> None:
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
    """One task's annotation: its reader reads, its drawer draws.

    A plain composer rather than an ABC — there is nothing to override. A kind composes
    the two it needs; new behaviour is a new reader or drawer here and one line in a kind.
    """

    def __init__(self, reader: Reader, drawer: Drawer) -> None:
        self._reader = reader
        self._drawer = drawer

    def annotate(self, view: SampleView, task: TaskView, outputs: Tensor, targets: Tensor, index: int) -> None:
        """Label batch element ``index``; ``outputs`` are the task's activated outputs."""
        truth = self._reader.read_target(_numpy(targets[index]))
        predicted = self._reader.read_output(_numpy(outputs[index]))
        self._drawer.annotate(view, task, truth, predicted)


def _numpy(tensor: Tensor) -> np.ndarray:
    array: np.ndarray = tensor.detach().cpu().float().numpy()
    return array
