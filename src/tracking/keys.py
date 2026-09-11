"""Structured metric identity; consumers never infer context from variable-length strings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from src.core import SEGMENT, SPLIT, Stage, validate_name


@dataclass(frozen=True, slots=True)
class MetricKey:
    """``stage[@split]/[task/]name`` — the one grammar every reported value is written and read in.

    The split is shown only when it is not the stage's own; a name may carry segments of
    its own (``f1/cat``) so a vector metric's leaves group under one family.
    """

    stage: Stage
    name: str
    split: str | None = None
    task: str | None = None

    def __post_init__(self) -> None:
        if self.task is not None:
            validate_name(self.task, label="Task")
        if self.split is not None:
            validate_name(self.split, label="Split")
        segments = self.name.split(SEGMENT)
        if any(not part or part.strip() != part or SPLIT in part for part in segments):
            raise ValueError(
                f"Metric names need nonblank {SEGMENT!r}-separated segments without {SPLIT!r}: {self.name!r}."
            )

    @classmethod
    def parse(cls, text: str) -> Self:
        """Read a key back; a first segment that is not a stage is outside the grammar."""
        head, *rest = text.split(SEGMENT)
        stage_text, _, split = head.partition(SPLIT)
        if not rest or stage_text not in Stage:
            raise ValueError(f"Not a metric key: {text!r}.")
        task = rest[0] if len(rest) > 1 else None
        return cls(Stage(stage_text), SEGMENT.join(rest[1:] if task else rest), split=split or None, task=task)

    @property
    def family(self) -> str:
        """Everything but the last segment: the graph a leaf belongs to."""
        return str(self).rsplit(SEGMENT, 1)[0]

    @property
    def leaf(self) -> str:
        return self.name.rsplit(SEGMENT, 1)[-1]

    def __str__(self) -> str:
        head = self.stage if self.split in (None, self.stage) else f"{self.stage}{SPLIT}{self.split}"
        return SEGMENT.join(segment for segment in (head, self.task, self.name) if segment)
