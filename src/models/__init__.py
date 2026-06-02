"""Models: backbone/head adapters and multi-head assembly.

Importing this package registers the built-in backbones and heads so they are
available by key in the registries.
"""

from src.models.assembly import CompositeModel, build_composite_model
from src.models.backbones import TimmBackbone
from src.models.heads import ConvHead, LinearHead
from src.models.registry import backbones, head_builders

__all__ = [
    "CompositeModel",
    "ConvHead",
    "LinearHead",
    "TimmBackbone",
    "backbones",
    "build_composite_model",
    "head_builders",
]
