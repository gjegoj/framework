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

    Which segment becomes the series follows from what the key is worth comparing:

    - ``train/label/ce`` → ``("label/ce", "train")``. The stage is the series, so
      train, val and test of one number share a graph.
    - ``val/label/f1/cat`` → ``("val/label/f1", "cat")``. A fourth segment appears
      only in the family a vector metric writes — its ``mean`` and one entry per
      class — and comparing the classes is what a per-class metric is for. Splitting
      by stage instead would draw forty graphs of one line on a forty-class run. The
      cost: train and val means then sit on separate graphs rather than overlaid.
    - ``lr/backbone`` → ``("lr", "backbone")``. A key that starts with no stage is a
      family whose leaves are the comparison — did the head move faster than the
      encoder.
    - ``epoch`` → ``("epoch", "value")``. Nothing to split, so the series is a constant.

    Returns:
        tuple[str, str]: The graph a value is drawn on, and the line within it.
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
