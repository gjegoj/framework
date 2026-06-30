"""Transforms: per-sample input augmentation and cross-sample batch transforms.

Custom per-sample augmentations are re-exported here so configs can reference them by the short
path (``_target_: src.transforms.Rotate90WithLabel``). The ``augmentations`` package pulls in only
albumentations + numpy, so this stays light.

Batch transforms are deliberately NOT re-exported: import ``src.transforms.batch`` directly so
that pulling the per-sample transform into the data layer does not drag in torchvision/tasks via
the batch package.
"""

from src.transforms.augmentations import (
    LabelAwareDualTransform,
    LabelAwareMixin,
    RandomBorderCropWithLabel,
    Rotate90WithLabel,
)

__all__ = [
    "LabelAwareDualTransform",
    "LabelAwareMixin",
    "RandomBorderCropWithLabel",
    "Rotate90WithLabel",
]
