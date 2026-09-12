"""Objectives that score a predicted region against the wanted one, rather than a pixel at a time."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from segmentation_models_pytorch.losses import (
    BINARY_MODE,
    MULTICLASS_MODE,
    MULTILABEL_MODE,
    DiceLoss,
    JaccardLoss,
    TverskyLoss,
)
from torch import nn

from src.core import Semantics
from src.losses.base import TorchLoss
from src.losses.registry import loss_registry

MODES: Mapping[Semantics, str] = {
    Semantics.BINARY: BINARY_MODE,
    Semantics.MULTICLASS: MULTICLASS_MODE,
    Semantics.MULTILABEL: MULTILABEL_MODE,
}
"""smp's own word for what a label means; the framework states it once, and this is where it is translated."""


class RegionLoss(TorchLoss):
    """An smp region loss, told in smp's dialect what its task's labels mean.

    The task settles the semantics, so a run never writes it: a binary mask scored by dice no longer
    has to remember that the library defaults to multiclass and would silently score a degenerate
    one-hot instead.
    """

    def __init__(self, semantics: Semantics | None = None, **options: Any) -> None:
        if semantics is None:
            raise ValueError(
                f"{type(self).__name__} scores a predicted region against labelled pixels, and this task's "
                "target carries no labels. Learn a number with mse or mae instead."
            )
        super().__init__(mode=MODES[semantics], **options)


@loss_registry.register("dice")
class Dice(RegionLoss):
    """Overlap of the predicted region with the wanted one — the segmentation staple."""

    module_type: ClassVar[type[nn.Module]] = DiceLoss


@loss_registry.register("iou")
class IntersectionOverUnion(RegionLoss):
    """The stricter overlap reading: what both agree on, over everything either claims."""

    module_type: ClassVar[type[nn.Module]] = JaccardLoss


@loss_registry.register("tversky")
class Tversky(RegionLoss):
    """Dice with the two kinds of mistake weighed apart: `alpha` false alarms, `beta` misses.

    Measured against smp 0.5.0 rather than read off the name: on a prediction that is all false alarms
    `alpha=0.9` costs 0.474 and `alpha=0.1` costs 0.091, and on one that is all misses the two swap. A
    run that wants recall raises `beta`.
    """

    module_type: ClassVar[type[nn.Module]] = TverskyLoss
