"""Criterion implementations (loss bricks).

Criteria operate on logits and return a ``LossResult`` (a backprop total plus
named components for logging). They register in the ``criteria`` registry so
objective strategies — and users — can select them by key.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torch import Tensor, nn

from src.core.entities import LossResult
from src.core.ports import Criterion
from src.core.registry import Registry

criteria: Registry[Criterion] = Registry("criterion")


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
        pw = torch.tensor(pos_weight, dtype=torch.float) if pos_weight is not None else None
        self._loss = nn.BCEWithLogitsLoss(pos_weight=pw, reduction=reduction)

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


@criteria.register("composite")
class CompositeCriterion(Criterion):
    """Weighted sum of several criteria sharing the same (logits, target).

    Lets a YAML ``loss:`` combine bricks, e.g. ``CE + Dice`` for segmentation::

        loss: {name: composite, terms: {cross_entropy: 1.0, dice: 1.0}}

    Each sub-criterion's components are forwarded for logging; the total is the
    weighted sum of the sub-totals.

    Parameters:
        terms (dict[str, float]): ``{criterion_key: weight}`` from the ``criteria`` registry.
    """

    def __init__(self, terms: dict[str, float]) -> None:
        super().__init__()
        if not terms:
            raise ValueError("CompositeCriterion needs at least one term.")
        self._weights = list(terms.values())
        self._criteria = nn.ModuleList(criteria.create(key) for key in terms)

    def forward(self, logits: Tensor, target: Tensor) -> LossResult:
        total = logits.new_zeros(())
        components: dict[str, Tensor] = {}
        for weight, criterion in zip(self._weights, self._criteria, strict=True):
            result = criterion(logits, target)
            total = total + weight * result.total
            components.update(result.components)
        return LossResult(total=total, components=components)
