"""A batch, a step and a run's tasks become the values ``src/visualization`` draws.

Two questions, and one class each side of them. *What did this task say about this sample* is read by a
``Reader``, whose answer depends on what the labels mean — one class, one score, or one score per label.
*How is that shown* is a ``Drawer``, which depends on where the decision was taken: once for the sample,
or at every pixel. A kind pairs the two, and because the pair is typed together, a reader can only be
composed with a drawer that draws what it reads.

Neither is declared anywhere. Both follow from facts the task already publishes — its semantics and the
shape it produces — so a kind of one's own is drawn without saying a word about drawing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_never

import numpy as np
import torch

from src.core import CELLS, ERROR, OVERLAP, Normalization, Semantics, class_name, require_tensor
from src.tasks import DECISION
from src.visualization import (
    Classification,
    Classifications,
    Image,
    Regression,
    SampleView,
    Score,
    Segmentation,
    SegmentationClass,
    Verdict,
)

if TYPE_CHECKING:
    from torch import Tensor

    from src.core import Batch, InputInfo, StepOutput
    from src.tasks import Task

NEGATIVE, POSITIVE = 0, 1
"""The two sides of a binary reading, in the order every vocabulary numbers them."""

# The scores themselves are computed here rather than borrowed from `metrics`: those objects are
# configured for a whole epoch — segmentation asks them for a per-class vector — so serving one sample
# would need a second configuration of the same thing. Only the words are shared, from `core`.


@dataclass(frozen=True, slots=True, eq=False)
class ClassPresence:
    """One class that holds, where it holds, and how strongly.

    ``where`` is boolean over the sample's positions: ``()``-shaped where the decision was taken once
    for the sample, ``[H, W]`` where it was taken at every pixel. That one array is what lets the
    drawer, rather than the reader, decide between a chip and a mask. ``confidence`` is ``None`` on the
    ground-truth side, which expressed none.
    """

    index: int
    where: np.ndarray
    confidence: float | None = None
    scored: bool = True
    """Whether this is a class the run is measured on, rather than the absence of one.

    A binary reading names both sides so that a confident "no" reads as an answer on a chip. Only the
    positive side is a class: it is the one ``BinaryJaccardIndex`` scores, and painting the negative
    one would cover the whole image in the colour of "not the thing".
    """


@dataclass(frozen=True, slots=True, eq=False)
class ClassReading:
    """Which classes hold. ``singular`` is what the semantics allow, not what this sample happened to show."""

    presences: tuple[ClassPresence, ...]
    singular: bool


@dataclass(frozen=True, slots=True, eq=False)
class ValueReading:
    """A number per position: ``()`` for a regressed scalar, ``[H, W]`` for a field of them."""

    values: np.ndarray


class Reader[R](ABC):
    """How one kind of label semantics reads a sample's prediction and its target, at any topology.

    ``scores`` is one sample's *activated* output, and the activation decides whether it still carries
    a class axis: multiclass and multilabel arrive as ``[C, *positions]``, while binary and continuous
    arrive as ``[*positions]`` — the class axis is dropped where it held one value. ``target`` arrives
    hard: indices, an indicator vector, or numbers.
    """

    @abstractmethod
    def read_output(self, scores: np.ndarray) -> R:
        """What the model said about this sample."""

    @abstractmethod
    def read_target(self, target: np.ndarray) -> R:
        """What is true of it."""


class Drawer[R](ABC):
    """How one topology turns a pair of readings into what a cell shows and what it scored."""

    @abstractmethod
    def draw(self, view: SampleView, task: Task, truth: R, predicted: R) -> None:
        """Write this task's two sides and its verdict into the view."""


@dataclass(frozen=True, slots=True)
class Annotator[R]:
    """A reader and the drawer for what it reads — the whole of how one kind appears on a page.

    Typed together, so the mismatch legacy had to refuse at runtime, a reading of one kind handed to a
    drawer of another, cannot be constructed at all.
    """

    reader: Reader[R]
    drawer: Drawer[R]

    def annotate(self, view: SampleView, task: Task, outputs: Tensor, targets: Tensor, index: int) -> None:
        """Draw batch element ``index``; ``outputs`` are this task's activated outputs for the batch."""
        self.drawer.draw(
            view, task, self.reader.read_target(_numpy(targets[index])), self.reader.read_output(_numpy(outputs[index]))
        )


type AnyAnnotator = Annotator[ClassReading] | Annotator[ValueReading]


def annotator_for(task: Task) -> AnyAnnotator | None:
    """How this task is drawn, read off what it already declares — or ``None`` where nothing draws it.

    Two facts decide it and neither is new: what its labels mean, and whether it decides at every pixel.
    The one pairing with no answer is a number per pixel, which is a heat map — a shape the display
    vocabulary has no label for yet, so a page leaves it off and says so rather than drawing it wrong.
    """
    reader: Reader[ClassReading]
    match task.semantics:
        case Semantics.MULTICLASS:
            reader = MulticlassReader()
        case Semantics.BINARY:
            reader = BinaryReader()
        case Semantics.MULTILABEL:
            reader = MultilabelReader()
        case None:
            return None if task.dense else Annotator(ValueReader(), NumberDrawer())
        case _:
            assert_never(task.semantics)
    return Annotator(reader, MaskDrawer() if task.dense else ChipDrawer())


class MulticlassReader(Reader[ClassReading]):
    """The highest-scoring class at each position; outputs arrive activated, so there is no softmax here."""

    def read_output(self, scores: np.ndarray) -> ClassReading:
        chosen = scores.argmax(axis=0)
        return ClassReading(
            presences=tuple(_presence(int(index), chosen == index, scores) for index in np.unique(chosen)),
            singular=True,
        )

    def read_target(self, target: np.ndarray) -> ClassReading:
        return ClassReading(
            presences=tuple(ClassPresence(int(index), target == index) for index in np.unique(target)),
            singular=True,
        )


class BinaryReader(Reader[ClassReading]):
    """One score per position, read against the line. There is no class axis to take a maximum over."""

    def read_output(self, scores: np.ndarray) -> ClassReading:
        # `scores` *is* the probability of the positive class — the activation dropped the one channel
        # — so both sides are scored from it, and a confident "no" reads as a confident answer.
        return ClassReading(presences=_sides(scores >= DECISION, np.stack([1.0 - scores, scores])), singular=True)

    def read_target(self, target: np.ndarray) -> ClassReading:
        return ClassReading(presences=_sides(target >= DECISION, None), singular=True)


class MultilabelReader(Reader[ClassReading]):
    """Independent scores: every class above the line holds, and any number of them may."""

    def read_output(self, scores: np.ndarray) -> ClassReading:
        positive = scores >= DECISION
        return ClassReading(
            presences=tuple(
                _presence(index, positive[index], scores) for index in range(positive.shape[0]) if positive[index].any()
            ),
            singular=False,
        )

    def read_target(self, target: np.ndarray) -> ClassReading:
        positive = target >= DECISION
        return ClassReading(
            presences=tuple(
                ClassPresence(index, positive[index]) for index in range(positive.shape[0]) if positive[index].any()
            ),
            singular=False,
        )


class ValueReader(Reader[ValueReading]):
    """The activation already collapsed the class axis, so what arrives is the number itself."""

    def read_output(self, scores: np.ndarray) -> ValueReading:
        return ValueReading(values=scores)

    def read_target(self, target: np.ndarray) -> ValueReading:
        return ValueReading(values=target)


class ChipDrawer(Drawer[ClassReading]):
    """One decision for the whole sample: two chips, matched by comparing what holds on each side."""

    def draw(self, view: SampleView, task: Task, truth: ClassReading, predicted: ClassReading) -> None:
        view.fields[(task.name, "gt")] = self._chips(task, truth)
        view.fields[(task.name, "pred")] = self._chips(task, predicted)
        # Set equality, so `correct` means *everything* matched: one class missing or one extra is a
        # miss, whatever the semantics allow.
        matched = {one.index for one in truth.presences} == {one.index for one in predicted.presences}
        view.verdicts[task.name] = Verdict(correct=matched)

    @staticmethod
    def _chips(task: Task, reading: ClassReading) -> Classification | Classifications:
        found = tuple(
            Classification(class_name(task.info.classes, one.index), confidence=one.confidence)
            for one in reading.presences
        )
        # A single-label reading holds exactly one class; an empty one draws nothing rather than
        # inventing a label for what the model did not say.
        return found[0] if reading.singular and found else Classifications(classifications=found)


class MaskDrawer(Drawer[ClassReading]):
    """One decision per pixel: a mask per class, scored by the overlap over the classes either side shows."""

    def draw(self, view: SampleView, task: Task, truth: ClassReading, predicted: ClassReading) -> None:
        view.fields[(task.name, "gt")] = self._masks(task, truth)
        view.fields[(task.name, "pred")] = self._masks(task, predicted)
        overlap = _mean_overlap(truth, predicted)
        view.verdicts[task.name] = Verdict(scores=() if overlap is None else (Score(OVERLAP, overlap),))

    @staticmethod
    def _masks(task: Task, reading: ClassReading) -> Segmentation:
        return Segmentation(
            classes=tuple(
                SegmentationClass(class_name(task.info.classes, one.index), one.where)
                for one in reading.presences
                if one.scored and one.where.any()
            )
        )


class NumberDrawer(Drawer[ValueReading]):
    """One number for the whole sample: two chips, and how far apart they are."""

    def draw(self, view: SampleView, task: Task, truth: ValueReading, predicted: ValueReading) -> None:
        true_value, predicted_value = _scalar(truth.values), _scalar(predicted.values)
        view.fields[(task.name, "gt")] = Regression(true_value)
        view.fields[(task.name, "pred")] = Regression(predicted_value)
        view.verdicts[task.name] = Verdict(scores=(Score(ERROR, abs(predicted_value - true_value)),))


@dataclass(frozen=True, slots=True)
class Gallery:
    """One run's tasks and the input a page draws them over, ready to turn a step into cells.

    Built once, when a run's facts are known, so nothing per-batch asks again what a task means or how
    to undo a normalisation.
    """

    tasks: Mapping[str, Task]
    annotators: Mapping[str, AnyAnnotator]
    input_name: str
    normalization: Normalization

    @classmethod
    def of(cls, tasks: Mapping[str, Task], input_name: str, normalization: Normalization) -> Gallery:
        drawn = {name: annotator for name, task in tasks.items() if (annotator := annotator_for(task)) is not None}
        return cls(tasks=tasks, annotators=drawn, input_name=input_name, normalization=normalization)

    @property
    def undrawable(self) -> tuple[str, ...]:
        """The tasks nothing draws, named so whoever built this can say which, and once.

        Derived rather than stored: two stored fields already say it, and a third could be built
        disagreeing with them.
        """
        return tuple(sorted(self.tasks.keys() - self.annotators.keys()))

    @property
    def classes(self) -> dict[str, tuple[str, ...]]:
        """Each task's whole vocabulary, so a class keeps its colour from one page of a run to the next."""
        return {name: named for name, task in self.tasks.items() if (named := vocabulary_of(task))}

    def views(self, batch: Batch, output: StepOutput, count: int) -> list[SampleView]:
        """The first ``count`` samples of this batch as cells: the image, and what each task said of it."""
        sources = _sources(batch, self.input_name, count)
        views = []
        for index, pixels in enumerate(_images(batch.inputs[self.input_name], self.normalization, count)):
            view = SampleView(image=Image(pixels=pixels, source=sources[index]))
            for name, annotator in self.annotators.items():
                predicted, targets = output.predictions.get(name), output.targets.get(name)
                if predicted is not None and targets is not None:
                    annotator.annotate(view, self.tasks[name], _tensor(predicted), _tensor(targets), index)
            views.append(view)
        return views


def vocabulary_of(task: Task) -> tuple[str, ...]:
    """The class names a page can show for this task, which is what anchors its colours.

    Declared where the target declares one. A binary task declares none — its two sides are named by
    the same rule that names a per-class metric leaf — and without them a page showing only one of the
    two would recolour it, which is the drift a fixed batch exists to make visible rather than cause.
    """
    if task.info.classes:
        return tuple(task.info.classes[index] for index in sorted(task.info.classes))
    if task.semantics is Semantics.BINARY:
        return (class_name(None, NEGATIVE), class_name(None, POSITIVE))
    return ()


def drawn_input(inputs: Mapping[str, InputInfo]) -> tuple[str, Normalization] | None:
    """Which input a page draws and how to undo what the run did to it, or ``None`` where none can be.

    Declaring the statistics is the whole test. A page shows an image as the file held it, which means
    undoing the normalisation the run applied, and an input that never said what it applied cannot be
    undone. Declaration order rather than a preference of ours, so the answer does not move between
    runs. Both halves come back together, because an input that passed the test has both.
    """
    for name, info in inputs.items():
        if info.normalization is not None:
            return name, info.normalization
    return None


def _images(values: object, normalization: Normalization, count: int) -> list[np.ndarray]:
    """Undo the run's own normalisation and lay the channels out the way a browser reads them."""
    images = _tensor(values)[:count].detach().cpu().float()
    # The statistics come from the encoder that declared the channels, so there is one of each per
    # channel by construction — the page never has to guess at a pairing the input already settled.
    mean = torch.tensor(normalization.mean).view(1, -1, 1, 1)
    std = torch.tensor(normalization.std).view(1, -1, 1, 1)
    # Rounded rather than truncated: `.byte()` cuts toward zero, which lands a level off the source on
    # 62 of 256 values — and checking an image against the original is the whole job here.
    restored = (images * std + mean).clamp(0.0, 1.0).mul(255).round().byte()
    if restored.shape[1] == 1:
        restored = restored.repeat(1, 3, 1, 1)
    return list(restored.permute(0, 2, 3, 1).numpy())


def _sources(batch: Batch, input_name: str, count: int) -> list[str | None]:
    """Where each sample came from, where the pipeline carried it — a path on the cell, not a guess."""
    cells = batch.metadata.get(CELLS)
    if not isinstance(cells, Sequence):
        return [None] * count
    return [_source_in(cells[index], input_name) if index < len(cells) else None for index in range(count)]


def _source_in(row: object, input_name: str) -> str | None:
    found = row.get(input_name) if isinstance(row, Mapping) else None
    return found if isinstance(found, str) else None


def _presence(index: int, where: np.ndarray, scores: np.ndarray, *, scored: bool = True) -> ClassPresence:
    """Confidence is the mean score where the class holds.

    At a whole-sample decision that is one number, which is what a chip shows. Over a region it is an
    average, and no drawer shows one yet — a mask is drawn by its shape. One expression rather than a
    branch on a topology the reader deliberately does not know: the average costs a pass over a region
    that was just read, and telling the two cases apart here would be the reader learning what it is
    read by.
    """
    return ClassPresence(index=index, where=where, confidence=float(scores[index][where].mean()), scored=scored)


def _sides(positive: np.ndarray, scores: np.ndarray | None) -> tuple[ClassPresence, ...]:
    """The two sides of a binary reading, each kept only where it holds anywhere.

    The negative side is drawn and not scored. Averaging it in was measured against the run's own
    metric and is a different quantity: on a sparse mask a model predicting nothing scored 0.48 here
    against the 0.0 ``BinaryJaccardIndex`` reports — the background term hides exactly the samples
    the page exists to find.
    """
    sides = ((NEGATIVE, ~positive, False), (POSITIVE, positive, True))
    return tuple(
        ClassPresence(index, where, scored=scored) if scores is None else _presence(index, where, scores, scored=scored)
        for index, where, scored in sides
        if where.any()
    )


def _mean_overlap(truth: ClassReading, predicted: ClassReading) -> float | None:
    """Averaged over the classes either side shows, and only over the ones the run is measured on.

    That makes it the same quantity the run's own chart reports: measured against torchmetrics, a
    multiclass sample matches its macro reading exactly, and a binary one now matches
    ``BinaryJaccardIndex`` instead of averaging the foreground with the background.

    ``None`` where nothing is left to measure — a sample on which neither side claims anything has no
    overlap, and a zero there would sort it as the worst thing on the page. Torchmetrics answers 0.0
    for that case; a per-sample reading is a different question from an epoch's.
    """
    true_masks = {one.index: one.where for one in truth.presences if one.scored}
    predicted_masks = {one.index: one.where for one in predicted.presences if one.scored}
    shown = sorted(set(true_masks) | set(predicted_masks))
    if not shown:
        return None
    empty = np.zeros(next(iter({**true_masks, **predicted_masks}.values())).shape, dtype=bool)
    both = [_overlap(true_masks.get(index, empty), predicted_masks.get(index, empty)) for index in shown]
    return float(np.mean(both))


def _overlap(left: np.ndarray, right: np.ndarray) -> float:
    """How much of what either side claimed, both did. Every index reaching here holds somewhere.

    That is what makes the union non-empty: an index is in scope only because a reading listed it,
    and every reader drops a presence that holds nowhere.
    """
    return int((left & right).sum()) / int((left | right).sum())


def _scalar(values: np.ndarray) -> float:
    """The single number a whole-sample reading holds, whether it arrived 0-d or as ``[1]``."""
    return float(values.reshape(-1)[0])


def _numpy(tensor: Tensor) -> np.ndarray:
    array: np.ndarray = tensor.detach().cpu().float().numpy()
    return array


def _tensor(values: object) -> Tensor:
    return require_tensor(values, name="a page's values")
