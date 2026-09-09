"""The models capability: implementations of the core ``Model`` port and its parts."""

from __future__ import annotations

from src.models.adapters import LoraAdapters, merge_adapters
from src.models.backbones import (
    HFTextBackbone,
    MultiEncoderBackbone,
    MultiViewBackbone,
    SmpBackbone,
    TimmBackbone,
    UltralyticsBackbone,
)
from src.models.checkpoints import load_weights
from src.models.composite import CompositeModel, TaskComponents
from src.models.distillation import DistilledModel, without_teachers
from src.models.heads import ConvHead, CosineHead, DetectHead, ExpandedHead, LinearHead

__all__ = [
    "CompositeModel",
    "ConvHead",
    "CosineHead",
    "DetectHead",
    "DistilledModel",
    "ExpandedHead",
    "HFTextBackbone",
    "LinearHead",
    "LoraAdapters",
    "MultiEncoderBackbone",
    "MultiViewBackbone",
    "SmpBackbone",
    "TaskComponents",
    "TimmBackbone",
    "UltralyticsBackbone",
    "load_weights",
    "merge_adapters",
    "without_teachers",
]
