"""Custom per-sample Albumentations augmentations (plugged into a Compose via ``_target_``).

Each augmentation is a standalone albumentations transform living in its own module;
``LabelAwareDualTransform`` (``base``) is the shared base for those that also rewrite a discrete
label. Importing names from here is a convenience — configs reference the concrete module path
(e.g. ``src.transforms.augmentations.rotate.Rotate90WithLabel``) directly.
"""

from src.transforms.augmentations.base import LabelAwareDualTransform
from src.transforms.augmentations.rotate import Rotate90WithLabel

__all__ = ["LabelAwareDualTransform", "Rotate90WithLabel"]
