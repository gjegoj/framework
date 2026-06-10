"""Backbone adapters. Importing this package registers built-in backbones."""

from src.models.backbones.embedding import EmbeddingBackbone
from src.models.backbones.smp import SmpBackbone
from src.models.backbones.timm import TimmBackbone
from src.models.registry import backbones

__all__ = ["EmbeddingBackbone", "SmpBackbone", "TimmBackbone", "backbones"]
