"""Networks: the contracts a run composes, the composite family, its heads and its backbone adapters.

Importing this package is what makes its names resolvable: every implementation below registers as its
module runs, so `model=resnet18` in a config finds `timm` here and nowhere else.
"""

from __future__ import annotations

from src.models.adapters import Adapter, LoraAdapter
from src.models.backbones import (
    HFTextBackbone,
    MultiEncoderBackbone,
    MultiViewBackbone,
    SmpBackbone,
    TimmBackbone,
)
from src.models.base import Backbone, HeadConnection, Model, Neck, Produces, ShapeAware
from src.models.composite import CompositeModel
from src.models.heads import ConvHead, CosineHead, LinearHead
from src.models.necks import Projector
from src.models.weights import load_weights

__all__ = [
    "Adapter",
    "Backbone",
    "CompositeModel",
    "ConvHead",
    "CosineHead",
    "HFTextBackbone",
    "HeadConnection",
    "LinearHead",
    "LoraAdapter",
    "Model",
    "MultiEncoderBackbone",
    "MultiViewBackbone",
    "Neck",
    "Produces",
    "Projector",
    "ShapeAware",
    "SmpBackbone",
    "TimmBackbone",
    "load_weights",
]
