"""Criterion implementations (loss bricks).

Criteria operate on logits and return a ``LossResult`` (a backprop total plus
named components for logging). They register in the ``criteria`` registry so
objective strategies — and users — can select them by key.
"""

from __future__ import annotations

from typing import Any, cast

import segmentation_models_pytorch as smp
import torch
from torch import Tensor, nn

from src.core.entities import LossResult
from src.core.instantiate import instantiate
from src.core.ports import Criterion
from src.losses.registry import criteria


class _SingleTermCriterion(Criterion):
    """Base for criteria that wrap one loss module and log it under one component.

    Subclasses build their ``nn.Module`` loss in ``__init__`` and hand it to
    ``super().__init__``; they set ``_component_name`` — the label the scalar is
    logged under (conventionally the registry key).

    Parameters:
        loss (nn.Module): Wrapped loss, called as ``loss(logits, target)``.
    """

    _component_name: str

    def __init__(self, loss: nn.Module) -> None:
        super().__init__()
        self._loss = loss

    def forward(self, logits: Tensor, target: Tensor) -> LossResult:
        value: Tensor = self._loss(logits, target)
        return LossResult(total=value, components={self._component_name: value})


@criteria.register("cross_entropy")
class CrossEntropyCriterion(_SingleTermCriterion):
    """Multiclass cross-entropy on logits ``[B, C]`` vs class indices ``[B]``.

    Parameters:
        label_smoothing (float): Smoothing in ``[0, 1)`` applied to the targets.
        weight (list[float] | None): Optional per-class rescaling weights.
        ignore_index (int): Target value ignored in the loss (default ``-100``).
    """

    _component_name = "cross_entropy"

    def __init__(
        self,
        label_smoothing: float = 0.0,
        weight: list[float] | None = None,
        ignore_index: int = -100,
    ) -> None:
        weight_tensor = torch.tensor(weight, dtype=torch.float) if weight is not None else None
        super().__init__(
            nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing, ignore_index=ignore_index)
        )


@criteria.register("bce")
class BCEWithLogitsCriterion(_SingleTermCriterion):
    """Binary / multilabel BCE on logits vs float targets.

    Operates on logits ``[B]`` or ``[B, C]`` vs float targets of the same shape.
    Use for binary (out=1) and multilabel (out=C) objectives.

    Parameters:
        pos_weight (list[float] | None): Per-class positive-class weight (length C).
        reduction (str): ``"mean"`` (default) / ``"sum"`` / ``"none"``.
    """

    _component_name = "bce"

    def __init__(self, pos_weight: list[float] | None = None, reduction: str = "mean") -> None:
        pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float) if pos_weight is not None else None
        super().__init__(nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction=reduction))


@criteria.register("mse")
class MSECriterion(_SingleTermCriterion):
    """Mean squared error on raw outputs vs float targets.

    Parameters:
        reduction (str): ``"mean"`` (default) / ``"sum"``.
    """

    _component_name = "mse"

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(nn.MSELoss(reduction=reduction))


@criteria.register("l1")
class L1Criterion(_SingleTermCriterion):
    """Mean absolute error on raw outputs vs float targets.

    Parameters:
        reduction (str): ``"mean"`` (default) / ``"sum"``.
    """

    _component_name = "l1"

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(nn.L1Loss(reduction=reduction))


@criteria.register("dice")
class DiceCriterion(_SingleTermCriterion):
    """Soft Dice loss (overlap-based) on logits — strong for segmentation.

    Wraps ``smp.losses.DiceLoss``; for multiclass it consumes ``[B, C, H, W]``
    logits vs ``[B, H, W]`` index targets.

    Parameters:
        mode (str): ``"multiclass"`` (default) / ``"multilabel"`` / ``"binary"``.
    """

    _component_name = "dice"

    def __init__(self, mode: str = "multiclass") -> None:
        super().__init__(smp.losses.DiceLoss(mode=mode))


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
