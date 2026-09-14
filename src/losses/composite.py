"""Several objectives on one task: their weighted sum, with every term still reported as itself."""

from __future__ import annotations

from collections.abc import Sequence
from functools import reduce
from typing import cast

from torch import Tensor, nn

from src.core import LossOutput, Representation
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
        self.reads = _one_reading(parts)
        self.reads_per_sample = _one_count(parts)
        self.reads_soft_targets = all(loss.reads_soft_targets for loss, _ in parts)

    def forward(self, outputs: Tensor, targets: Tensor) -> LossOutput:
        weighted = [
            weight * cast(Loss, part)(outputs, targets) for part, weight in zip(self.parts, self.weights, strict=True)
        ]
        return reduce(lambda total, term: total + term, weighted)


def _one_count(parts: Sequence[tuple[Loss, float]]) -> int:
    """How many answers of a sample these terms agree the output holds; they read one tensor, so they must.

    The same argument as ``_one_reading`` below, about the other half of what that tensor is: a term
    reading one answer per sample and a term reading two cannot both be right about the same rows, and
    the one that is wrong is comparing a sample against somebody else's answer.
    """
    counted = {loss.reads_per_sample for loss, _ in parts}
    if len(counted) != 1:
        raise ValueError(
            f"These objectives disagree about how many answers one sample gives: {sorted(counted)}. They "
            f"are handed the same rows, so one of them is reading another sample's answer as this one's."
        )
    return counted.pop()


def _one_reading(parts: Sequence[tuple[Loss, float]]) -> Representation:
    """What these terms agree the head's numbers are; they all read the same tensor, so they must agree.

    A head produces one thing. Terms disagreeing about what that is could not both be right, and the
    one that is wrong would train and report a plausible number — so the disagreement is the refusal,
    stated here rather than left to whichever term happens to be checked against the head.
    """
    readings = {loss.reads for loss, _ in parts}
    if len(readings) > 1:
        named = ", ".join(f"{loss.log_name} reads {loss.reads}" for loss, _ in parts)
        raise ValueError(
            f"These objectives are summed over one and the same output and disagree about what it "
            f"holds: {named}. One head answers with one thing; declare terms that read it."
        )
    return readings.pop()
