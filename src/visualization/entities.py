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

    Scalar because that is what this framework produces: ``ContinuousObjective``
    activates through ``squeeze_single_output`` or ``expectation_over``, and both
    collapse a head's output to one value before any consumer sees it. A head
    predicting several quantities wants a plural container beside this one, the
    way ``Classifications`` sits beside ``Classification`` — see the backlog.

    It carries no error field. A draft did, and the error then appeared twice on
    a cell: once as a delta on the chip and once in the verdict. Ground truth and
    prediction sit side by side as chips, so the direction of a miss is already
    readable; how far it missed belongs to the verdict, which is the one thing
    that can be filtered on.
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

    Named as the framework names it — ``iou``, ``mae``, the keys
    ``metric_registry`` holds and the presets declare — so the page and the
    progress table do not call one quantity two things.

    It carries no direction. A draft did, on the argument that
    ``higher_is_better`` is the framework's own word for one; but the page filters
    with a two-handle range, which was chosen precisely because a band needs no
    preferred end, and nothing else ever read the field. A number nobody reads is
    not a fact, it is a claim — and this one was asserted by the annotator, while
    ``JaccardIndex.higher_is_better`` is in fact ``None``.

    If a direction-aware view ever arrives — worst-first ordering, a marked end on
    a slider — it should read ``metric_registry``, which owns the metric, rather
    than re-deriving it here.
    """

    name: str
    value: float


@dataclass(frozen=True, slots=True)
class Verdict:
    """How one task judged this sample, structured so no consumer parses a string.

    ``correct`` is a whole-match answer: ``True`` only when everything the task
    predicted matches everything that is true, ``False`` when any of it does not,
    and ``None`` where the task has no binary notion of rightness at all — a
    regression misses by an amount, it is not wrong. Samples of the last kind stay
    visible under every correct/wrong filter rather than being judged by a
    threshold nobody chose.

    ``scores`` are the measured numbers: an IoU, an error magnitude, anything a
    task can put on a scale. They replaced a free-text summary, which had to be
    written by the annotator, read by a human, and re-parsed by nothing at all —
    the number could not be filtered on, and the same value ended up printed twice.

    Plural because a task can measure itself more than one way — a segmentation
    sample has an IoU and a Dice — and because the page keys its sliders by task
    *and* metric either way, so one score and several cost the same code.
    """

    correct: bool | None = None
    scores: tuple[Score, ...] = ()


@dataclass(slots=True)
class SampleView:
    """One sample projected for display: what it shows and what was said about it.

    The framework-agnostic middle layer between annotators (torch) and a renderer
    (HTML) — plain dataclasses over numpy and stdlib, so either side can change without
    the other noticing. The label names are FiftyOne's, read from its API; the library
    itself is not a dependency, being a database-backed application whose labels are
    documents that cannot exist outside a ``Dataset``.

    ``media`` is keyed by input alias, as ``Batch.inputs`` is, because a sample may have
    more than one — a CLIP-style run has an image beside a caption, and a grid that drew
    only the first would halve what the run is about.

    ``fields`` is keyed by a structural ``(task, kind)`` tuple, never a glued string:
    the flat form a browser needs is built once at the HTML boundary and never read
    back, so a task named with an underscore cannot scramble it.

    Mutable on purpose: each task's annotator writes its fields and verdict in turn, and
    the renderer reads the finished view.
    """

    media: dict[str, Media] = field(default_factory=dict)
    fields: dict[tuple[str, Kind], Label] = field(default_factory=dict)
    verdicts: dict[str, Verdict] = field(default_factory=dict)
