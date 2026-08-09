"""Backbone adapters, one module per source library; importing this registers them all."""

from __future__ import annotations

from src.models.backbones.hf import HFTextBackbone
from src.models.backbones.multi import MultiEncoderBackbone, MultiViewBackbone
from src.models.backbones.smp import SmpBackbone
from src.models.backbones.timm import TimmBackbone

__all__ = [
    "HFTextBackbone",
    "MultiEncoderBackbone",
    "MultiViewBackbone",
    "SmpBackbone",
    "TimmBackbone",
]
