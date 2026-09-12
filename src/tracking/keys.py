"""Structured metric identity; consumers never infer context from variable-length strings."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Self

from src.core import SEGMENT, Stage, validate_name

MEAN = "mean"
"""The leaf a per-class family is read at: the one line among them that stands for the whole."""


def series(task: str | None, name: str) -> str:
    """What train, val and test each say about the same measurement — the key without its stage.

    Composed here rather than at its callers, because the measured fact (which way a metric is
    better, which row of a table it is) belongs to the measurement and not to the stage.
    """
    return SEGMENT.join(part for part in (task, name) if part)


@dataclass(frozen=True, slots=True)
class MetricKey:
    """``stage/[task/]name`` — the one grammar every reported value is written and read in.

    A name may carry segments of its own (``f1/cat``) so a vector metric's leaves group under one
    family. The task is a field rather than a segment to count, which is what lets a composed name
    mean leaves and nothing else.
    """

    stage: Stage
    name: str
    task: str | None = None

    def __post_init__(self) -> None:
        if self.task is not None:
            validate_name(self.task, label="Task")
        if any(not part or part.strip() != part for part in self.name.split(SEGMENT)):
            raise ValueError(f"Metric names need nonblank {SEGMENT!r}-separated segments: {self.name!r}.")

    @classmethod
    def parse(cls, text: str) -> Self:
        """Read a key back; a first segment that is not a stage is outside the grammar."""
        stage, *rest = text.split(SEGMENT)
        if not rest or stage not in Stage:
            raise ValueError(f"Not a metric key: {text!r}.")
        task = rest[0] if len(rest) > 1 else None
        return cls(Stage(stage), SEGMENT.join(rest[1:] if task else rest), task=task)

    @classmethod
    def headline(cls, text: str) -> Self | None:
        """The at-a-glance reading a logged key contributes to: itself, its family, or nothing.

        One rule, because a summary table and a progress table ask the same question of the same keys.
        A number stands for itself; a family stands at its ``mean`` and each class below it is detail;
        anything that is not a key of ours — ``epoch``, ``lr/backbone`` — is not a measurement to show.
        """
        try:
            key = cls.parse(text)
        except ValueError:
            return None
        if SEGMENT not in key.name:
            return key
        family, leaf = key.name.rsplit(SEGMENT, 1)
        return replace(key, name=family) if leaf == MEAN else None

    @property
    def series(self) -> str:
        """What every stage says about this same measurement; see :func:`series`."""
        return series(self.task, self.name)

    @property
    def family(self) -> str:
        """Everything but the last segment: the graph a leaf belongs to."""
        return str(self).rsplit(SEGMENT, 1)[0]

    @property
    def leaf(self) -> str:
        return self.name.rsplit(SEGMENT, 1)[-1]

    def __str__(self) -> str:
        return SEGMENT.join(segment for segment in (self.stage, self.task, self.name) if segment)
