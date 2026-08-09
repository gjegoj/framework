"""Augmentations that create supervision as well as perturbing an image."""

from __future__ import annotations

from src.transforms.augmentations.border_crop import SIDES, RandomBorderCrop
from src.transforms.augmentations.rotate import QUARTER_TURNS, Rotate90

__all__ = ["QUARTER_TURNS", "SIDES", "RandomBorderCrop", "Rotate90"]
