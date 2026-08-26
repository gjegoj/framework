"""A quarter-turn that also tells you how far it turned."""

from __future__ import annotations

from typing import Any

import albumentations as A
import numpy as np

QUARTER_TURNS = 4
"""Distinct 90° rotations — and therefore the number of rotation classes."""


class Rotate90(A.CustomTransformsApplyMixin, A.DualTransform):
    """Turn by a random multiple of 90° counter-clockwise and advance the bound label.

    One draw turns the image, any mask, and the label as ``(label + turns) % 4``, so a
    folder of upright photographs becomes a balanced four-class rotation task. The label is
    expected to hold the image's *current* rotation class — zero for an unrotated dataset.
    Bind it with ``AlbumentationsTransform(label_targets=["angle"])``.

    Parameters:
        p (float): Probability of turning at all.
    """

    def get_params(self) -> dict[str, int]:
        return {"turns": self.py_random.randint(0, QUARTER_TURNS - 1)}

    def apply(self, img: np.ndarray, turns: int = 0, **params: Any) -> np.ndarray:
        return np.ascontiguousarray(np.rot90(img, turns))

    def apply_to_mask(self, mask: np.ndarray, turns: int = 0, **params: Any) -> np.ndarray:
        return np.ascontiguousarray(np.rot90(mask, turns))

    def apply_to_label(self, label: int, turns: int = 0, **params: Any) -> int:
        return (label + turns) % QUARTER_TURNS
