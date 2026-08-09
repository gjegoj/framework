"""Segmentation criteria."""

from __future__ import annotations

from typing import Any, ClassVar

from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss, TverskyLoss

from src.losses.base import WrappedCriterion
from src.losses.registry import criterion_registry


@criterion_registry.register("dice")
class DiceCriterion(WrappedCriterion):
    """Dice loss on raw logits, via smp.

    Multiclass by default (``[B, C, H, W]`` logits vs ``[B, H, W]`` index
    masks); ``mode="binary"``/``"multilabel"`` follow smp's conventions.

    Parameters:
        mode (str): smp dice mode: ``"multiclass"``, ``"binary"``, or
            ``"multilabel"``.
        **kwargs: Forwarded verbatim to ``smp.losses.DiceLoss``
            (``smooth``, ``classes``, ``ignore_index``, ...).
    """

    part_name: ClassVar[str] = "dice"

    def __init__(self, mode: str = "multiclass", **kwargs: Any) -> None:
        super().__init__(DiceLoss(mode=mode, **kwargs))


@criterion_registry.register("iou")
class IoUCriterion(WrappedCriterion):
    """IoU loss on raw logits, via smp.

    Multiclass by default (``[B, C, H, W]`` logits vs ``[B, H, W]`` index
    masks); ``mode="binary"``/``"multilabel"`` follow smp's conventions.

    Parameters:
        mode (str): smp iou mode: ``"multiclass"``, ``"binary"``, or
            ``"multilabel"``.
        **kwargs: Forwarded verbatim to ``smp.losses.JaccardLoss``
            (``smooth``, ``classes``, ``ignore_index``, ...).
    """

    part_name: ClassVar[str] = "iou"

    def __init__(self, mode: str = "multiclass", **kwargs: Any) -> None:
        super().__init__(JaccardLoss(mode=mode, **kwargs))


@criterion_registry.register("tversky")
class TverskyCriterion(WrappedCriterion):
    """Tversky loss on raw logits, via smp.

    Multiclass by default (``[B, C, H, W]`` logits vs ``[B, H, W]`` index
    masks); ``mode="binary"``/``"multilabel"`` follow smp's conventions.

    Parameters:
        mode (str): smp tversky mode: ``"multiclass"``, ``"binary"``, or
            ``"multilabel"``.
        **kwargs: Forwarded verbatim to ``smp.losses.TverskyLoss``
            (``smooth``, ``classes``, ``ignore_index``, ...).
    """

    part_name: ClassVar[str] = "tversky"

    def __init__(self, mode: str = "multiclass", **kwargs: Any) -> None:
        super().__init__(TverskyLoss(mode=mode, **kwargs))
