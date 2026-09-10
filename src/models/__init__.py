"""Networks: the contracts a run composes, the composite family, its heads and its backbone adapters.

Importing this package is what makes its names resolvable: every implementation below registers as its
module runs, so `model=resnet18` in a config finds `timm` here and nowhere else.
"""

from __future__ import annotations

from src.models.backbones import SmpBackbone, TimmBackbone
from src.models.base import Backbone, HeadConnection, Model, ShapeAware
from src.models.composite import CompositeModel
from src.models.heads import ConvHead, LinearHead

__all__ = [
    "Backbone",
    "CompositeModel",
    "ConvHead",
    "HeadConnection",
    "LinearHead",
    "Model",
    "ShapeAware",
    "SmpBackbone",
    "TimmBackbone",
]
