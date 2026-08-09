"""Arrived weights: the family-agnostic loading, and what each backbone family adds to it."""

from __future__ import annotations

from src.models.backbones.checkpoints.loading import load_arrived_weights
from src.models.backbones.checkpoints.smp import (
    SMP_HEAD_PREFIXES,
    replace_last_projection,
    transplanted_segmentation_head,
)
from src.models.backbones.checkpoints.timm import classifier_prefixes, transplanted_classifier

__all__ = [
    "SMP_HEAD_PREFIXES",
    "classifier_prefixes",
    "load_arrived_weights",
    "replace_last_projection",
    "transplanted_classifier",
    "transplanted_segmentation_head",
]
