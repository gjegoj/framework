"""Backbone adapters, one module per library; importing this registers every one of them.

``multiview`` and ``multiencoder`` name no library: each is declared around others and is still a
backbone, because each is about how a sample is read — one draws it over every view, the other routes
it to a tower apiece. What reads features rather than a sample is a neck, and those live beside this.
"""

from __future__ import annotations

from src.models.backbones.hf import HFTextBackbone
from src.models.backbones.multiencoder import MultiEncoderBackbone
from src.models.backbones.multiview import MultiViewBackbone
from src.models.backbones.smp import SmpBackbone
from src.models.backbones.timm import TimmBackbone

__all__ = [
    "HFTextBackbone",
    "MultiEncoderBackbone",
    "MultiViewBackbone",
    "SmpBackbone",
    "TimmBackbone",
]
