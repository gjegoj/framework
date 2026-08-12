"""A border crop that is always worth learning from, and says when it happened."""

from __future__ import annotations

from typing import Any

import albumentations as A

SIDES = ("left", "right", "top", "bottom")
"""The borders a crop may trim, in a fixed order so a draw is reproducible.

A crop that leaves nothing is impossible by construction: the parent rejects
opposite limits summing above 1, so no widening here can meet its counterpart.
"""


class RandomBorderCrop(A.CustomTransformsApplyMixin, A.RandomCropFromBorders):
    """Trim a random strip from each border and mark the sample as cropped.

    ``min_crop`` is the reason this exists rather than the parent alone: a
    uniform draw may trim two pixels, and a sample cropped by two pixels teaches
    a model nothing while being labelled as cropped. When no side reaches the
    threshold, one eligible side is widened to it — a correction on top of the
    parent's draw, so the parent keeps owning the distribution.

    The other class comes from the transform *not* applying, so the dataset's
    own labels have to be the negative class already.

    Bind the column it rewrites with
    ``AlbumentationsTransform(label_targets=["was_cropped"])``.

    Parameters:
        crop_left (float): Largest fraction of the width trimmed from the left.
        crop_right (float): Largest fraction of the width trimmed from the right.
        crop_top (float): Largest fraction of the height trimmed from the top.
        crop_bottom (float): Largest fraction of the height trimmed from the bottom.
        min_crop (float): Fraction at least one side must reach; ``0`` asks for
            no guarantee. It cannot exceed every per-side maximum, or no side
            could ever satisfy it.
        applied_label (int | str): The raw value the bound column takes when the
            crop applies — encoding runs *after* the transforms, so write the class
            as the table writes it (``cropped`` beside
            ``classes: {0: intact, 1: cropped}``).
        p (float): Probability of cropping at all.
    """

    def __init__(
        self,
        crop_left: float = 0.1,
        crop_right: float = 0.1,
        crop_top: float = 0.1,
        crop_bottom: float = 0.1,
        min_crop: float = 0.0,
        applied_label: int | str = 1,
        p: float = 1.0,
    ) -> None:
        super().__init__(
            crop_left=crop_left,
            crop_right=crop_right,
            crop_top=crop_top,
            crop_bottom=crop_bottom,
            p=p,
        )
        largest = max(crop_left, crop_right, crop_top, crop_bottom)
        if min_crop > largest:
            raise ValueError(
                f"min_crop ({min_crop}) exceeds every per-side maximum (largest is {largest}), "
                f"so no side could ever reach it."
            )
        self.min_crop = min_crop
        self.applied_label = applied_label

    def apply_to_label(self, label: Any, **params: Any) -> int | str:
        return self.applied_label

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, tuple[int, int, int, int]]:
        taken = super().get_params_dependent_on_data(params, data)
        if self.min_crop <= 0.0 or max(self.applied_config.values()) >= self.min_crop:
            return taken

        # Widen one side in the record the parent just filled, then read the crop back
        # out of it: the parent keeps owning both the draw and what it reports.
        limits = {f"crop_{side}": getattr(self, f"crop_{side}") for side in SIDES}
        widened = self.py_random.choice([side for side, limit in limits.items() if limit >= self.min_crop])
        self.applied_config[widened] = self.py_random.uniform(self.min_crop, limits[widened])

        height, width = params["shape"][:2]
        fractions = self.applied_config
        return {
            "crop_coords": (
                int(fractions["crop_left"] * width),
                int(fractions["crop_top"] * height),
                width - int(fractions["crop_right"] * width),
                height - int(fractions["crop_bottom"] * height),
            )
        }
