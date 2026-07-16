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
    weighted sum of the sub-totals. Terms keep their config label and are readable
    by it — ``criterion["focal"]`` — which is how ``criterion_schedule`` addresses a
    term's parameter (``parameter: focal.gamma``).

    Parameters:
        losses (dict[str, float | dict]): Term specs keyed by label.
    """

    def __init__(self, losses: dict[str, float | dict[str, Any]]) -> None:
        super().__init__()
        if not losses:
            raise ValueError("WeightedSumCriterion needs at least one loss.")
        weights: dict[str, float] = {}
        terms: dict[str, Criterion] = {}
        for key, spec in losses.items():
            if isinstance(spec, (int, float)):
                weights[key] = float(spec)
                terms[key] = criteria.create(key)
            else:
                params = dict(spec)
                weights[key] = float(params.pop("weight", 1.0))
                if "_target_" in params:
                    terms[key] = instantiate({"_target_": params.pop("_target_"), **params})
                else:
                    terms[key] = criteria.create(key, **params)
        self._weights = weights
        self._criteria = nn.ModuleDict(terms)

    def __getitem__(self, label: str) -> Criterion:
        """Return the term registered under ``label`` (the key from the config)."""
        if label not in self._criteria:
            raise KeyError(f"WeightedSumCriterion has no term {label!r}; terms: {sorted(self._criteria.keys())}.")
        return cast(Criterion, self._criteria[label])

    def keys(self) -> list[str]:
        """Term labels, in config order (the addressable dot-path segments)."""
        return list(self._criteria.keys())

    def forward(self, logits: Tensor, target: Tensor) -> LossResult:
        total = logits.new_zeros(())
        components: dict[str, Tensor] = {}
        for label, criterion in self._criteria.items():
            result = cast(Criterion, criterion)(logits, target)  # ModuleDict iteration erases the element type
            total = total + self._weights[label] * result.total
            components.update(result.components)
        return LossResult(total=total, components=components)
