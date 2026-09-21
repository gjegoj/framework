"""Backbone adapters, one module per library; importing this registers every one of them.

``multiview`` and ``projector`` name no library: each is declared around another backbone — one draws
it over every view of a sample, the other brings one of its streams to a declared width — so they are
here beside the families rather than above them.
"""

from __future__ import annotations

from src.models.backbones.hf import HFTextBackbone
from src.models.backbones.multiencoder import MultiEncoderBackbone
from src.models.backbones.multiview import MultiViewBackbone
from src.models.backbones.projector import ProjectorBackbone
from src.models.backbones.smp import SmpBackbone
from src.models.backbones.timm import TimmBackbone

__all__ = [
    "HFTextBackbone",
    "MultiEncoderBackbone",
    "MultiViewBackbone",
    "ProjectorBackbone",
    "SmpBackbone",
    "TimmBackbone",
]
