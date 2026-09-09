"""A kind of task declared outside the framework: what a user's own class looks like."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from src.losses import FocalCriterion
from src.tasks import Classification

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.core import Criterion, TaskFacts


class FocalClassification(Classification):
    """Classification learned with a focal loss and judged by accuracy alone.

    Reachable as ``kind: {_target_: tests.support.kinds.FocalClassification}`` — the
    extension path the framework promises: one class, no edit to the framework.
    """

    default_metrics: ClassVar[Mapping[str, Mapping[str, Any]]] = {"accuracy": {"name": "accuracy"}}

    def loss(self, facts: TaskFacts, width: int) -> Criterion:
        return FocalCriterion(gamma=2.0)
