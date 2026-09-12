"""What a stage does to a sample, and what a callback does to a batch.

Three roles, and the folders say which is which: ``albumentations`` runs the chain a stage declares,
``augmentations`` holds the operations that go inside such a chain, and ``batch`` holds the transforms
that need a whole collated batch. The contracts they answer are in ``base``.
"""

from __future__ import annotations

from src.transforms.albumentations import AlbumentationsTransform
from src.transforms.augmentations import RandomBorderCrop, Rotate90
from src.transforms.base import AnswersTask, BatchTransform, GeometryAware, SampleTransform
from src.transforms.batch import CutMix, MixUp

__all__ = [
    "AlbumentationsTransform",
    "AnswersTask",
    "BatchTransform",
    "CutMix",
    "GeometryAware",
    "MixUp",
    "RandomBorderCrop",
    "Rotate90",
    "SampleTransform",
]
