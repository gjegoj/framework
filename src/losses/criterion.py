"""Criterion implementations (loss bricks).

Criteria operate on logits and return a ``LossResult`` (a backprop total plus
named components for logging). They register in the ``criteria`` registry so
objective strategies — and users — can select them by key.
"""

from __future__ import annotations

from typing import Any

import segmentation_models_pytorch as smp
import torch
from torch import Tensor, nn

from src.core.entities import LossResult
from src.core.instantiate import instantiate
from src.core.ports import Criterion
from src.losses.registry import criteria


@criteria.register("cross_entropy")
class CrossEntropyCriterion(Criterion):
    """Multiclass cross-entropy on logits ``[B, C]`` vs class indices ``[B]``.

    Parameters:
        label_smoothing (float): Smoothing in ``[0, 1)`` applied to the targets.
        weight (list[float] | None): Optional per-class rescaling weights.
        ignore_index (int): Target value ignored in the loss (default ``-100``).
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        weight: list[float] | None = None,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        weight_tensor = torch.tensor(weight, dtype=torch.float) if weight is not None else None
        self._loss = nn.CrossEntropyLoss(
            weight=weight_tensor,
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
        )

    def forward(self, logits: Tensor, target: Tensor) -> LossResult:
        value: Tensor = self._loss(logits, target)
        return LossResult(total=value, components={"cross_entropy": value})


@criteria.register("bce")
class BCEWithLogitsCriterion(Criterion):
    """Binary / multilabel BCE on logits vs float targets.

    Operates on logits ``[B]`` or ``[B, C]`` vs float targets of the same shape.
    Use for binary (out=1) and multilabel (out=C) objectives.

    Parameters:
        pos_weight (list[float] | None): Per-class positive-class weight (length C).
        reduction (str): ``"mean"`` (default) / ``"sum"`` / ``"none"``.
    """

    def __init__(
        self,
        pos_weight: list[float] | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float) if pos_weight is not None else None
        self._loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction=reduction)

    def forward(self, logits: Tensor, target: Tensor) -> LossResult:
        value: Tensor = self._loss(logits, target)
        return LossResult(total=value, components={"bce": value})


@criteria.register("mse")
class MSECriterion(Criterion):
    """Mean squared error on raw outputs vs float targets.

    Parameters:
        reduction (str): ``"mean"`` (default) / ``"sum"``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self._loss = nn.MSELoss(reduction=reduction)

    def forward(self, logits: Tensor, target: Tensor) -> LossResult:
        value: Tensor = self._loss(logits, target)
        return LossResult(total=value, components={"mse": value})


@criteria.register("l1")
class L1Criterion(Criterion):
    """Mean absolute error on raw outputs vs float targets.

    Parameters:
        reduction (str): ``"mean"`` (default) / ``"sum"``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self._loss = nn.L1Loss(reduction=reduction)

    def forward(self, logits: Tensor, target: Tensor) -> LossResult:
        value: Tensor = self._loss(logits, target)
        return LossResult(total=value, components={"l1": value})


@criteria.register("dice")
class DiceCriterion(Criterion):
    """Soft Dice loss (overlap-based) on logits — strong for segmentation.

    Wraps ``smp.losses.DiceLoss``; for multiclass it consumes ``[B, C, H, W]``
    logits vs ``[B, H, W]`` index targets.

    Parameters:
        mode (str): ``"multiclass"`` (default) / ``"multilabel"`` / ``"binary"``.
    """

    def __init__(self, mode: str = "multiclass") -> None:
        super().__init__()
        self._loss = smp.losses.DiceLoss(mode=mode)

    def forward(self, logits: Tensor, target: Tensor) -> LossResult:
        value: Tensor = self._loss(logits, target)
        return LossResult(total=value, components={"dice": value})


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
            raise ValueError("CompositeCriterion needs at least one loss.")
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
            result = criterion(logits, target)
            total = total + weight * result.total
            components.update(result.components)
        return LossResult(total=total, components=components)
