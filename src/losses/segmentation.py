"""Segmentation criteria: overlap-based losses over dense logits.

``segmentation_models_pytorch`` is imported at module level deliberately — this module is
only imported via the package ``__init__`` alongside the smp-backed backbones, so the
dependency is already a hard one for any segmentation experiment.
"""

from __future__ import annotations

from typing import Any

import segmentation_models_pytorch as smp

from src.losses.base import SingleTermCriterion
from src.losses.registry import criteria


@criteria.register("dice")
class DiceCriterion(SingleTermCriterion):
    """Soft Dice loss (overlap-based) on logits — strong for segmentation.

    Wraps ``smp.losses.DiceLoss``; for multiclass it consumes ``[B, C, H, W]``
    logits vs ``[B, H, W]`` index targets.

    Parameters:
        mode (str): ``"multiclass"`` (default) / ``"multilabel"`` / ``"binary"``.
        **kwargs: Forwarded verbatim to ``smp.losses.DiceLoss``
            (``smooth``, ``eps``, ``log_loss``, ``ignore_index``, ``classes``, ...).
    """

    component_name = "dice"

    def __init__(self, mode: str = "multiclass", **kwargs: Any) -> None:
        super().__init__(smp.losses.DiceLoss(mode=mode, **kwargs))
