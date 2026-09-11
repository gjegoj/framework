"""Several objectives on one task: their weighted sum, with every term still reported as itself."""

from __future__ import annotations

from collections.abc import Sequence
from functools import reduce
from typing import cast

from torch import Tensor, nn

from src.core import LossOutput
from src.losses.base import Loss


class WeightedSum(Loss):
    """What a task learns when its declaration lists more than one loss.

    A weight changes the objective, never the number a report shows: each term is logged as itself and
    its weighted share separately, so a run can see that dice is 0.4 and contributes 0.2.
    """

    def __init__(self, parts: Sequence[tuple[Loss, float]]) -> None:
        super().__init__()
        if not parts:
            raise ValueError("A weighted sum needs at least one loss.")
        self.parts = nn.ModuleList(loss for loss, _ in parts)
        self.weights = [float(weight) for _, weight in parts]

    def forward(self, outputs: Tensor, targets: Tensor) -> LossOutput:
        weighted = [
            weight * cast(Loss, part)(outputs, targets) for part, weight in zip(self.parts, self.weights, strict=True)
        ]
        return reduce(lambda total, term: total + term, weighted)
