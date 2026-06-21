"""Rotation augmentation that generates the rotation class online.

``Rotate90WithLabel`` rotates the image (and any mask) by a random multiple of 90°
counter-clockwise and bumps the bound rotation label by the same number of quarter-turns.
Paired with the framework's encoder contract — where ``TargetEncoder.load`` already yields the
class *index* before the transform runs — a single all-0° dataset becomes a balanced 4-class
rotation task without duplicating any images.

Used via ``_target_`` in a transforms config (no registration needed).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.transforms.augmentations.base import LabelAwareDualTransform

# Distinct 90° rotations (0/90/180/270) — also the number of rotation classes.
_QUARTER_TURNS = 4


class Rotate90WithLabel(LabelAwareDualTransform):
    """Rotate by a random multiple of 90° CCW and bump the bound rotation class (0/90/180/270).

    The quarter-turn count ``k`` is sampled once per call and applied consistently to the image,
    any mask, and the rotation label (``(index + k) % 4``), so the visual rotation and its class
    always agree.
    """

    def get_params(self) -> dict[str, int]:
        return {"k": self.py_random.randint(0, _QUARTER_TURNS - 1)}

    def apply(self, img: np.ndarray, k: int = 0, **params: Any) -> np.ndarray:
        return np.ascontiguousarray(np.rot90(img, k))

    def apply_to_mask(self, mask: np.ndarray, k: int = 0, **params: Any) -> np.ndarray:
        return np.ascontiguousarray(np.rot90(mask, k))

    def apply_to_label(self, label: int, k: int = 0, **params: Any) -> int:
        return (label + k) % _QUARTER_TURNS
