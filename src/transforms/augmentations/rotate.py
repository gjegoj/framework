"""A quarter-turn that also tells you how far it turned."""

from __future__ import annotations

from typing import Any

import albumentations as A
import numpy as np

QUARTER_TURNS = 4
"""Distinct 90° rotations — and therefore the number of rotation classes."""


class Rotate90(A.CustomTransformsApplyMixin, A.DualTransform):
    """Turn by a random multiple of 90° counter-clockwise and advance the bound label.

    The turn count is sampled once and applied to the image, to any mask, and to
    the label as ``(label + turns) % 4`` — so the picture and its class cannot
    drift apart. A folder of upright photographs therefore becomes a balanced
    four-class rotation task without a single file being duplicated, and the
    class count follows from the data as it does for any other task.

    The label is expected to hold the image's *current* rotation class, which
    for an unrotated dataset is zero.

    Bind the column it rewrites with
    ``AlbumentationsTransform(label_targets=["angle"])``.

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
