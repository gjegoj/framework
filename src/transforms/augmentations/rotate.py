"""A quarter-turn that also says how far it turned."""

from __future__ import annotations

from typing import Any, override

import albumentations as A
import numpy as np

QUARTER_TURNS = 4
"""Distinct 90° rotations, and therefore the number of classes a rotation target carries."""


class Rotate90(A.CustomTransformsApplyMixin, A.DualTransform):
    """Turn by a random multiple of 90° counter-clockwise, and advance the target by as much.

    Parameters:
        task: The task this answers; its target holds the image's current turn, in quarters.
        p: How often it turns at all; a draw of zero turns is one of the four outcomes either way.
    """

    def __init__(self, task: str, p: float = 1.0) -> None:
        super().__init__(p=p)
        self.task = task

    @override
    def get_params(self) -> dict[str, int]:
        return {"turns": self.py_random.randint(0, QUARTER_TURNS - 1)}

    @override
    def apply(self, img: np.ndarray, turns: int = 0, **params: Any) -> np.ndarray:
        # Contiguous, because a view with negative strides is one torch refuses to make a tensor of.
        return np.ascontiguousarray(np.rot90(img, turns))

    @override
    def apply_to_mask(self, mask: np.ndarray, turns: int = 0, **params: Any) -> np.ndarray:
        return np.ascontiguousarray(np.rot90(mask, turns))

    def apply_to_label(self, label: Any, turns: int = 0, **params: Any) -> int:
        return (int(label) + turns) % QUARTER_TURNS
