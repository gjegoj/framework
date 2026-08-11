"""Augmentations that create supervision as well as perturbing an image."""

from __future__ import annotations

from src.transforms.augmentations.border_crop import SIDES, RandomBorderCrop
from src.transforms.augmentations.rotate import QUARTER_TURNS, Rotate90
from src.transforms.augmentations.warm_region import COOLEST, WARMEST, MaskedPlanckianJitter

__all__ = [
    "COOLEST",
    "QUARTER_TURNS",
    "SIDES",
    "WARMEST",
    "MaskedPlanckianJitter",
    "RandomBorderCrop",
    "Rotate90",
]
