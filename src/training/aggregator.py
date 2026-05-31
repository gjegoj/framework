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
        total = torch.zeros(1, device=next(iter(losses.values())).total.device).squeeze()
        components: dict[str, torch.Tensor] = {}
        for task_name, result in losses.items():
            w = weights.get(task_name, 1.0)
            total = total + w * result.total
            for component_name, value in result.components.items():
                components[f"{task_name}/{component_name}"] = value
        return LossResult(total=total, components=components)
