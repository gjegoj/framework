"""Augmentations of our own, one module each; they go *inside* a chain, not in place of one.

What a stage declares is an ``AlbumentationsTransform``; what it declares inside it is a list, and
these are entries in that list. Both of them answer a task rather than leaving its answer alone —
their draw is the supervision — which is the contract ``AnswersTask`` states and the pipeline reads.
"""

from __future__ import annotations

from src.transforms.augmentations.border_crop import SIDES, RandomBorderCrop
from src.transforms.augmentations.rotate import QUARTER_TURNS, Rotate90

__all__ = ["QUARTER_TURNS", "SIDES", "RandomBorderCrop", "Rotate90"]
