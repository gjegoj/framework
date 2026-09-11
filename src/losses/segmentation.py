"""Objectives that score a predicted region against the wanted one, rather than a pixel at a time."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss, TverskyLoss
from torch import nn

from src.losses.base import TorchLoss
from src.losses.registry import loss_registry

MULTICLASS = "multiclass"
"""smp's own name for one class per pixel; a binary or multilabel run says so in its declaration."""


@loss_registry.register("dice")
class Dice(TorchLoss):
    """Overlap of the predicted region with the wanted one — the segmentation staple."""

    module_type: ClassVar[type[nn.Module]] = DiceLoss
    defaults: ClassVar[Mapping[str, Any]] = {"mode": MULTICLASS}


@loss_registry.register("iou")
class IntersectionOverUnion(TorchLoss):
    """The stricter overlap reading: what both agree on, over everything either claims."""

    module_type: ClassVar[type[nn.Module]] = JaccardLoss
    defaults: ClassVar[Mapping[str, Any]] = {"mode": MULTICLASS}


@loss_registry.register("tversky")
class Tversky(TorchLoss):
    """Dice with the two kinds of mistake weighed apart: `alpha` misses, `beta` false alarms."""

    module_type: ClassVar[type[nn.Module]] = TverskyLoss
    defaults: ClassVar[Mapping[str, Any]] = {"mode": MULTICLASS}
