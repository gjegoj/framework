"""Batch transforms: cross-sample mixing on the collated batch.

Importing this package populates the ``batch_transforms`` registry.
"""

from src.transforms.batch.mixup import CutMix, MixUp
from src.transforms.batch.mosaic import Mosaic
from src.transforms.batch.registry import batch_transforms
from src.transforms.batch.spec import BatchTransform, TargetSpec

__all__ = ["BatchTransform", "CutMix", "MixUp", "Mosaic", "TargetSpec", "batch_transforms"]
