"""Visualization IR: one sample projected for display, in FiftyOne's vocabulary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

type Kind = Literal["gt", "pred"]

KINDS: tuple[Kind, ...] = ("gt", "pred")
"""Every field kind, in reading order — what is true first, what was predicted second."""


@dataclass(frozen=True, slots=True, eq=False)
class Image:
    """Pixels ready to display, and where they came from if anywhere.

    ``eq=False`` because an ndarray field breaks the generated ``__eq__`` — the
    same reason ``TaskComponents`` opts out, and nothing here compares media.

    Parameters:
        pixels (np.ndarray): ``[H, W, 3]`` uint8 RGB, already denormalised.
        source (str | None): The path or URL the row named, when it named one;
            a link or a copy-to-clipboard pill on the cell.
    """

    pixels: np.ndarray
    source: str | None = None


@dataclass(frozen=True, slots=True)
class Text:
    """A readable string shown beside its sample — a caption, a prompt, a document.

    Taken from the row rather than from the tensor on purpose: a text input
    reaches the model as ``input_ids``, and the tokenizer that produced them is
    not something a callback holds. The row still has the words.
    """

    text: str


type Media = Image | Text
"""What a sample shows. Keyed by input alias, because a sample may have several."""


@dataclass(frozen=True, slots=True)
class Classification:
    """One predicted or true class; ``confidence`` only where a model expressed one."""

    label: str
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class Classifications:
    """A multilabel set: every active class, each with its own confidence."""

    classifications: tuple[Classification, ...] = ()


@dataclass(frozen=True, slots=True)
class Regression:
    """One regressed number — FiftyOne's shape exactly.

    Scalar because that is what this framework produces: ``ContinuousObjective`` collapses
    a head's output to one value. It carries no error field: the direction of a miss is
    readable from the two chips, and how far belongs to the verdict, which can be filtered on.
    """

    value: float
    confidence: float | None = None


@dataclass(frozen=True, slots=True, eq=False)
class SegmentationClass:
    """One class's boolean mask at display resolution."""

    name: str
    mask: np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class Segmentation:
    """A segmentation label: the classes present in this sample, one mask each.

    Per class rather than FiftyOne's single integer map, because a dense
    multilabel task lets classes overlap — which one map cannot express — and
    because the sidebar switches classes on and off individually.
    """

    classes: tuple[SegmentationClass, ...] = ()


type Label = Classification | Classifications | Regression | Segmentation


@dataclass(frozen=True, slots=True)
class Score:
    """A number a task measured on one sample.

    Named as the framework names it — ``iou``, ``mae`` — so the page and the progress
    table call one quantity one thing. It carries no direction: the page filters with a
    two-handle range, and a direction-aware view should read ``metric_registry``.
    """

    name: str
    value: float


@dataclass(frozen=True, slots=True)
class Verdict:
    """How one task scored this sample, structured so no consumer parses a string.

    ``correct`` is a whole-match answer: ``True`` only when everything predicted matches,
    ``False`` when any of it does not, ``None`` where the task has no binary notion of
    rightness (a regression misses by an amount). ``scores`` are the measured numbers — an
    IoU, an error — plural because a task can measure itself more than one way.
    """

    correct: bool | None = None
    scores: tuple[Score, ...] = ()


@dataclass(slots=True)
class SampleView:
    """One sample projected for display: what it shows and what was said about it.

    The framework-agnostic layer between annotators (torch) and a renderer (HTML): plain
    dataclasses over numpy and stdlib. Label names are FiftyOne's; the library itself is
    not a dependency. ``media`` is keyed by input alias, as ``Batch.inputs`` is; ``fields``
    by a structural ``(task, kind)`` tuple, never a glued string. Mutable on purpose: each
    task's annotator writes its fields and verdict in turn.
    """

    media: dict[str, Media] = field(default_factory=dict)
    fields: dict[tuple[str, Kind], Label] = field(default_factory=dict)
    verdicts: dict[str, Verdict] = field(default_factory=dict)
