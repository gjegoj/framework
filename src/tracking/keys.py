"""Structured metric identity; consumers never infer context from variable-length strings."""

from __future__ import annotations

from dataclasses import dataclass

from src.core import Stage
from src.core.entities import validate_name


@dataclass(frozen=True, slots=True)
class MetricKey:
    stage: Stage
    name: str
    split: str | None = None
    task: str | None = None

    def __post_init__(self) -> None:
        if self.task is not None:
            validate_name(self.task)
        if self.split is not None and (
            not self.split or self.split == "_" or "/" in self.split or self.split.strip() != self.split
        ):
            raise ValueError("split must be nonblank, without '/' or reserved '_'.")
        if not self.name or any(not part or part.strip() != part for part in self.name.split("/")):
            raise ValueError("Metric names require nonblank path segments.")

    def __str__(self) -> str:
        return f"{self.stage.value}/{self.split or '_'}/{self.task or '_'}/{self.name}"
