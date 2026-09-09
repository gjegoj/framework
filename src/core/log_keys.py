"""The single owner of the log-key grammar ``{stage}/{task}/{leaf}``: composed here, parsed here, once."""

from __future__ import annotations

from typing import Final, NamedTuple

from src.core.taxonomy import Stage

SEPARATOR: Final = "/"
"""Joins key segments; also what ``Loss.scoped`` namespaces parts with."""

TOTAL_LOSS: Final = "loss"
"""Leaf of the total-loss key — the value monitors and schedulers watch."""

MEAN: Final = "mean"
"""Leaf of a vector metric's average, beside its per-class leaves."""

PER_CLASS_SEGMENTS: Final = 3
"""Segments after the stage once a metric has per-class leaves: ``{task}/{metric}/{class}``.

A scalar metric and a loss part have two (``{task}/{leaf}``) and a total has one,
so this counts exactly the family a vector metric writes — nothing needs to know
what the value means to recognise it.
"""

STAGES: Final = frozenset(Stage)
"""The stage tokens a key may start with — for consumers that classify keys.

``Stage`` is a ``StrEnum``, so a plain first segment compares equal to its
member ("val" in this set is True).
"""


def join(*segments: str) -> str:
    """Compose a log key from segments: ``join("val", "label", "ce")`` is ``"val/label/ce"``."""
    return SEPARATOR.join(segments)


def total_loss(stage: Stage) -> str:
    """The stage's total-loss key, e.g. ``"train/loss"``."""
    return join(stage, TOTAL_LOSS)


class LogKey(NamedTuple):
    """A log key taken apart once: its stage, when it starts with one, and the segments after it.

    The one place the grammar is read back, so no consumer splits a string by hand: a
    tracker's title and series, a summary's headline names and a progress table's rows
    all ask this value what a key is.
    """

    stage: Stage | None
    path: tuple[str, ...]

    @property
    def per_class(self) -> bool:
        """Whether the key is a vector metric's leaf — ``{task}/{metric}/{class}`` after the stage."""
        return self.stage is not None and len(self.path) >= PER_CLASS_SEGMENTS

    @property
    def leaf(self) -> str:
        return self.path[-1]

    @property
    def is_mean(self) -> bool:
        return self.path[-1] == MEAN

    @property
    def family(self) -> str:
        """The key without its leaf — what a per-class leaf's siblings share."""
        prefix = () if self.stage is None else (str(self.stage),)
        return join(*prefix, *self.path[:-1])

    @property
    def rest(self) -> str:
        """The key without its stage."""
        return join(*self.path)


def parse(key: str) -> LogKey:
    """Take a key apart: a first segment that is a stage is the stage, everything else is the path."""
    first, _, tail = key.partition(SEPARATOR)
    if first in STAGES:
        return LogKey(Stage(first), tuple(tail.split(SEPARATOR)) if tail else ())
    return LogKey(None, tuple(key.split(SEPARATOR)))
