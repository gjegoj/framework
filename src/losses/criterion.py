"""Criterion implementations (loss bricks).

Criteria operate on logits and return a ``LossResult`` (a backprop total plus
named components for logging). They register in the ``criteria`` registry so
objective strategies — and users — can select them by key.
"""

from __future__ import annotations

from torch import Tensor, nn

from src.core.entities import LossResult
from src.core.ports import Criterion
from src.core.registry import Registry

criteria: Registry[Criterion] = Registry("criterion")


@criteria.register("cross_entropy")
class CrossEntropyCriterion(Criterion):
    """Multiclass cross-entropy on logits ``[B, C]`` vs class indices ``[B]``."""

    def __init__(self) -> None:
        super().__init__()
        self._loss = nn.CrossEntropyLoss()

    def forward(self, logits: Tensor, target: Tensor) -> LossResult:
        value: Tensor = self._loss(logits, target)
        return LossResult(total=value, components={"cross_entropy": value})
