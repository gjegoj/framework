"""The single owner of the log-key grammar ``{stage}/{task}/{leaf}``: composed here, parsed here."""

from __future__ import annotations

from typing import Final

from src.core.taxonomy import Stage

SEPARATOR: Final = "/"
"""Joins key segments; also what ``Loss.scoped`` namespaces parts with."""

TOTAL_LOSS: Final = "loss"
"""Leaf of the total-loss key — the value monitors and schedulers watch."""

MEAN: Final = "mean"
"""Leaf of a vector metric's average, beside its per-class leaves."""

_PER_CLASS_SEGMENTS: Final = 3
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


def split_for_tracker(key: str) -> tuple[str, str]:
    """A key as a tracker's ``(title, series)`` — one graph per title, one line per series.

    - ``train/label/ce`` → ``("label/ce", "train")``: stages of one number share a graph.
    - ``val/label/f1/cat`` → ``("val/label/f1", "cat")``: a per-class family compares its
      classes on one graph, at the cost of train and val means sitting apart.
    - ``lr/backbone`` → ``("lr", "backbone")``: no stage, so the leaves are the comparison.
    - ``epoch`` → ``("epoch", "value")``.
    """
    stage, separator, rest = key.partition(SEPARATOR)
    if not separator:
        return key, "value"
    if stage not in STAGES:
        family, _, leaf = key.rpartition(SEPARATOR)
        return family, leaf
    if len(rest.split(SEPARATOR)) >= _PER_CLASS_SEGMENTS:
        graph, _, leaf = key.rpartition(SEPARATOR)
        return graph, leaf
    return rest, stage
