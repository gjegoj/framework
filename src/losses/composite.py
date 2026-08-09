"""Combining several criteria into one, when a task is judged on more than one thing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from src.core.entities import Loss
from src.core.ports import Criterion

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor


class WeightedSumCriterion(Criterion):
    """Several criteria on the same output, added with weights.

    Each part keeps its own name in the result, so a run logs the terms
    separately and their movements stay readable — a total that stops falling
    says much less than seeing which term stopped falling. Two parts sharing a
    name is refused by ``Loss`` rather than silently merged.

    The parts are held in a ``ModuleList``, so buffers a criterion registers —
    class values, class weights — follow the model across devices.

    Parameters:
        parts (Sequence[tuple[Criterion, float]]): Each criterion with its weight.
    """

    def __init__(self, parts: Sequence[tuple[Criterion, float]]) -> None:
        super().__init__()
        if not parts:
            raise ValueError("WeightedSumCriterion needs at least one part.")
        self._parts = nn.ModuleList([criterion for criterion, _ in parts])
        self._weights = [float(weight) for _, weight in parts]

    def forward(self, logits: Tensor, target: Tensor) -> Loss:
        return Loss.sum(weight * part(logits, target) for part, weight in zip(self._parts, self._weights, strict=True))
