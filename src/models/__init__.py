"""The models capability: implementations of the core ``Model`` port and its parts."""

from __future__ import annotations

from src.models.adapters import LoraAdapters, merge_adapters
from src.models.backbones.hf import HFTextBackbone
from src.models.backbones.multi import MultiEncoderBackbone, MultiViewBackbone
from src.models.backbones.smp import SmpBackbone
from src.models.backbones.timm import TimmBackbone
from src.models.composite import CompositeModel, TaskComponents
from src.models.distillation import DistilledModel, without_teachers
from src.models.heads import ConvHead, CosineHead, ExpandedHead, IdentityHead, LinearHead, WrappedHead
from src.models.yolo import YoloModel

__all__ = [
    "CompositeModel",
    "ConvHead",
    "CosineHead",
    "DistilledModel",
    "ExpandedHead",
    "HFTextBackbone",
    "IdentityHead",
    "LinearHead",
    "LoraAdapters",
    "MultiEncoderBackbone",
    "MultiViewBackbone",
    "SmpBackbone",
    "TaskComponents",
    "TimmBackbone",
    "WrappedHead",
    "YoloModel",
    "merge_adapters",
    "without_teachers",
]
