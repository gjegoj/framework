"""What a page is given: one sample projected for display.

Plain values over numpy and the standard library — no tensors, no notion of a task or a model. This is
the seam the rest of this package is written against, and the reason it can be lifted out whole: every
name below is something a viewer can see, and nothing above it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

type Side = Literal["gt", "pred"]
"""Which side of a cell something belongs to, in reading order — what is true, then what was said.

Not ``Kind``: this framework spells a task's kind that way in every experiment file it ships, and one
word for two things is the thing this package exists to avoid. The two values stay the words a
computer-vision reader already has, and the stylesheet keys off them.
"""

SIDES: tuple[Side, ...] = ("gt", "pred")


@dataclass(frozen=True, slots=True, eq=False)
class Image:
    """Pixels ready to show, and where they came from if anywhere.

    ``eq=False`` because an array field breaks the generated ``__eq__``, and nothing compares pictures.

    Parameters:
        pixels: ``[H, W, 3]`` uint8 RGB, already back in the colours a viewer expects.
        source: The path or URL the sample came from, when there is one — a link on the cell.
    """

    pixels: np.ndarray
    source: str | None = None


@dataclass(frozen=True, slots=True)
class Classification:
    """One class, with the confidence behind it where the side that said it expressed one."""

    label: str
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class Classifications:
    """Several classes at once, each with its own confidence — what a multilabel answer looks like."""

    classifications: tuple[Classification, ...] = ()


@dataclass(frozen=True, slots=True)
class Regression:
    """One number.

    It carries no error: which way a prediction missed is readable from the two chips, and how far
    is a score, which is the thing a page can filter on.
    """

    value: float


@dataclass(frozen=True, slots=True, eq=False)
class SegmentationClass:
    """One class's boolean mask, at the resolution it was predicted."""

    name: str
    mask: np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class Segmentation:
    """The classes present in one sample, a mask each.

    One mask per class rather than a single map of indices: classes may overlap where a dense task is
    multilabel, which one map cannot say, and the sidebar switches them on and off one at a time.
    """

    classes: tuple[SegmentationClass, ...] = ()


type Label = Classification | Classifications | Regression | Segmentation
"""Everything a side of a cell can say. Every arm of it is drawn; see ``renderers.py``."""


@dataclass(frozen=True, slots=True)
class Score:
    """A number measured on one sample, named as the run names it — ``iou``, ``mae``.

    No direction: the page filters with a two-handle band, which needs none, and a low overlap and a
    high error are the same complaint said two ways.
    """

    name: str
    value: float


@dataclass(frozen=True, slots=True)
class Verdict:
    """How one task scored one sample, structured so nothing downstream parses a string.

    ``correct`` answers the whole match — ``True`` when everything predicted agrees, ``False`` when
    any of it does not, ``None`` where the task has no yes-or-no notion of right (a number misses by
    an amount). ``scores`` are what was measured, plural because a task may be measured several ways.
    """

    correct: bool | None = None
    scores: tuple[Score, ...] = ()


@dataclass(slots=True)
class SampleView:
    """One cell: the picture, what each task said about it on each side, and how each one scored.

    ``fields`` is keyed by a structural ``(task, side)`` pair rather than a glued string, so nothing
    here has to split back what something else joined. Mutable on purpose: whoever builds a view
    fills in one task at a time.
    """

    picture: Image
    fields: dict[tuple[str, Side], Label] = field(default_factory=dict)
    verdicts: dict[str, Verdict] = field(default_factory=dict)
