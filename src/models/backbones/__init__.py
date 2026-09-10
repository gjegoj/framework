"""Backbone adapters, one module per library; importing this registers every one of them."""

from __future__ import annotations

from src.models.backbones.smp import SmpBackbone
from src.models.backbones.timm import TimmBackbone

__all__ = ["SmpBackbone", "TimmBackbone"]
