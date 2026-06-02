"""Backbone adapters. Importing this package registers built-in backbones."""

from src.models.backbones.smp import SmpBackbone
from src.models.backbones.timm import TimmBackbone
from src.models.registry import backbones

__all__ = ["SmpBackbone", "TimmBackbone", "backbones"]
