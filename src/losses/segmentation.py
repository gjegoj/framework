"""Segmentation criteria — overlap measures on raw logits, via smp.

All three take smp's own ``mode``, and it means the same thing in each: ``"multiclass"``
by default (``[B, C, H, W]`` logits against ``[B, H, W]`` index masks), with ``"binary"``
and ``"multilabel"`` following smp's conventions. Said once here rather than in each
class, where it was three copies of one sentence and the only thing that differed
between them was the name of the loss being wrapped.

Every other knob of the wrapped loss — ``smooth``, ``classes``, ``ignore_index`` —
forwards verbatim through ``**kwargs``, as it does for every wrapped criterion.
"""

from __future__ import annotations

from typing import Any, ClassVar

from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss, TverskyLoss

from src.losses.base import WrappedCriterion
from src.losses.registry import criterion_registry


@criterion_registry.register("dice")
class DiceCriterion(WrappedCriterion):
    """Twice the overlap over the summed areas — the F1 of a mask, as a loss.

    Parameters:
        mode (str): smp's mode; see the module docstring.
        **kwargs: Forwarded verbatim to ``smp.losses.DiceLoss``.
    """

    part_name: ClassVar[str] = "dice"

    def __init__(self, mode: str = "multiclass", **kwargs: Any) -> None:
        super().__init__(DiceLoss(mode=mode, **kwargs))


@criterion_registry.register("iou")
class IoUCriterion(WrappedCriterion):
    """The overlap over the union — the same quantity the ``iou`` metric reports.

    Parameters:
        mode (str): smp's mode; see the module docstring.
        **kwargs: Forwarded verbatim to ``smp.losses.JaccardLoss``.
    """

    part_name: ClassVar[str] = "iou"

    def __init__(self, mode: str = "multiclass", **kwargs: Any) -> None:
        super().__init__(JaccardLoss(mode=mode, **kwargs))


@criterion_registry.register("tversky")
class TverskyCriterion(WrappedCriterion):
    """Dice with the two error kinds weighted apart — for a target the model keeps missing.

    ``alpha`` and ``beta`` trade false positives against false negatives; equal, it *is*
    Dice.

    Parameters:
        mode (str): smp's mode; see the module docstring.
        **kwargs: Forwarded verbatim to ``smp.losses.TverskyLoss`` (``alpha``, ``beta``, ...).
    """

    part_name: ClassVar[str] = "tversky"

    def __init__(self, mode: str = "multiclass", **kwargs: Any) -> None:
        super().__init__(TverskyLoss(mode=mode, **kwargs))
