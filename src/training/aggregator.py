"""LossAggregator: combines per-task losses into a single training objective."""

from __future__ import annotations

import torch

from src.core.entities import LossResult
from src.core.ports import LossAggregator


class WeightedSumAggregator(LossAggregator):
    """Weighted sum of per-task total losses.

    total = Σ weight_i * task_i.total

    Each task's components are forwarded into the result under
    ``"<task_name>/<component_name>"`` keys for granular logging.
    """

    def combine(self, losses: dict[str, LossResult], weights: dict[str, float]) -> LossResult:
        # Match the dtype/device of the first task's loss (mirrors ``logits.new_zeros(())``
        # used in WeightedSumCriterion) — one zero-scalar idiom across the loss layer.
        total = next(iter(losses.values())).total.new_zeros(())
        components: dict[str, torch.Tensor] = {}
        for task_name, result in losses.items():
            weight = weights.get(task_name, 1.0)
            total = total + weight * result.total
            for component_name, value in result.components.items():
                components[f"{task_name}/{component_name}"] = value
        return LossResult(total=total, components=components)
