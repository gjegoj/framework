"""Backbone adapters, one module per library; importing this registers every one of them.

``multiview`` is the one that names no library: it draws another backbone over every view of a sample,
so it is here beside the families rather than above them.
"""

from __future__ import annotations

from src.models.backbones.hf import HFTextBackbone
from src.models.backbones.multiencoder import MultiEncoderBackbone
from src.models.backbones.multiview import MultiViewBackbone
from src.models.backbones.smp import SmpBackbone
from src.models.backbones.timm import TimmBackbone

__all__ = ["HFTextBackbone", "MultiEncoderBackbone", "MultiViewBackbone", "SmpBackbone", "TimmBackbone"]
