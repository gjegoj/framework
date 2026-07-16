"""Composite criteria: combine several criteria over the same (logits, target)."""

from __future__ import annotations

from typing import Any, cast

from torch import Tensor, nn

from src.core.entities import LossResult
from src.core.instantiate import instantiate
from src.core.ports import Criterion
from src.losses.registry import criteria


@criteria.register("weighted_sum")
class WeightedSumCriterion(Criterion):
    """Weighted sum of several criteria sharing the same (logits, target).

    Two term formats are supported (and can be mixed)::

        # Simple: key is the criteria registry key, value is the weight.
        loss: {name: weighted_sum, losses: {cross_entropy: 1.0, dice: 2.0}}

        # Parameterised: dict with ``weight`` plus any criterion kwargs.
        # Use ``_target_`` to bypass the registry and instantiate any class.
        loss:
          name: weighted_sum
          losses:
            cross_entropy: 1.0
            dice:
              weight: 2.0
              smooth: 1.0e-5
              mode: multiclass
            focal:
              weight: 10.0
              _target_: segmentation_models_pytorch.losses.FocalLoss
              gamma: 2.0

    Each sub-criterion's components are forwarded for logging; the total is the
    weighted sum of the sub-totals.

    Parameters:
        losses (dict[str, float | dict]): Term specs keyed by label.
    """

    def __init__(self, losses: dict[str, float | dict[str, Any]]) -> None:
        super().__init__()
        if not losses:
            raise ValueError("WeightedSumCriterion needs at least one loss.")
        weights: list[float] = []
        criterion_list: list[Criterion] = []
        for key, spec in losses.items():
            if isinstance(spec, (int, float)):
                weights.append(float(spec))
                criterion_list.append(criteria.create(key))
            else:
                params = dict(spec)
                weights.append(float(params.pop("weight", 1.0)))
                if "_target_" in params:
                    criterion_list.append(instantiate({"_target_": params.pop("_target_"), **params}))
                else:
                    criterion_list.append(criteria.create(key, **params))
        self._weights = weights
        self._criteria = nn.ModuleList(criterion_list)

    def forward(self, logits: Tensor, target: Tensor) -> LossResult:
        total = logits.new_zeros(())
        components: dict[str, Tensor] = {}
        for weight, criterion in zip(self._weights, self._criteria, strict=True):
            result = cast(Criterion, criterion)(logits, target)  # ModuleList iteration erases the element type
            total = total + weight * result.total
            components.update(result.components)
        return LossResult(total=total, components=components)
