"""Border-crop augmentation that guarantees a minimum crop and marks the sample online.

``RandomBorderCropWithLabel`` mixes the framework's ``LabelAwareMixin`` into Albumentations'
``RandomCropFromBorders`` (trim a random strip from each border — no resize, the output is
smaller) and adds two things:

* ``min_crop_threshold`` forces at least one border to be cropped by at least that fraction, so a
  uniform draw can never degenerate into an (almost) no-op crop.
* whenever the crop is applied it rewrites the bound discrete label to ``1``, the same online-label
  idea as ``Rotate90WithLabel`` — turning "an unaltered image vs. a border-cropped one" into a
  supervised binary signal without duplicating any images.

The per-border maxima and the crop-sum bounds (``crop_left + crop_right <= 1`` etc.) are validated
by the crop base; only the threshold invariant is checked here. Used via ``_target_`` in a
transforms config (no registration needed).
"""

from __future__ import annotations

from typing import Any

import albumentations as A

from src.transforms.augmentations.base import LabelAwareMixin

# The four borders, ordered for stable iteration; the strings are the per-side limit keys below.
_SIDES = ("left", "right", "top", "bottom")


class RandomBorderCropWithLabel(LabelAwareMixin, A.RandomCropFromBorders):
    """Crop a random fraction from each border (one side guaranteed >= a threshold) and set the label to 1.

    Behaves like ``RandomCropFromBorders`` (per-border maxima ``crop_left``/``crop_right``/
    ``crop_top``/``crop_bottom``); when every uniform draw lands below ``min_crop_threshold`` one
    eligible side is resampled into ``[min_crop_threshold, its maximum]`` so the guarantee always
    holds. The bound label (``label_key``) is rewritten to ``1`` on every application.

    Parameters:
        crop_left (float): Max fraction of width croppable from the left.
        crop_right (float): Max fraction of width croppable from the right.
        crop_top (float): Max fraction of height croppable from the top.
        crop_bottom (float): Max fraction of height croppable from the bottom.
        min_crop_threshold (float): Lower bound the strongest crop must reach on at least one side;
            ``0.0`` disables the guarantee. Must not exceed the largest per-side maximum.
        label_key (str): Data key of the label to set to ``1`` — the task's ``target`` column.
        p (float): Probability of applying the transform.
    """

    def __init__(
        self,
        crop_left: float = 0.1,
        crop_right: float = 0.1,
        crop_top: float = 0.1,
        crop_bottom: float = 0.1,
        min_crop_threshold: float = 0.0,
        label_key: str = "label",
        p: float = 1.0,
    ) -> None:
        self.label_key = label_key  # set before super().__init__ so _set_keys can register it
        super().__init__(
            crop_left=crop_left,
            crop_right=crop_right,
            crop_top=crop_top,
            crop_bottom=crop_bottom,
            p=p,
        )
        largest_side_limit = max(crop_left, crop_right, crop_top, crop_bottom)
        if min_crop_threshold > largest_side_limit:
            raise ValueError(
                f"min_crop_threshold ({min_crop_threshold}) cannot exceed the largest per-side "
                f"crop limit ({largest_side_limit})"
            )
        self.min_crop_threshold = min_crop_threshold

    def apply_to_label(self, label: Any, **params: Any) -> int:
        """Mark the sample as cropped — the label becomes ``1`` whenever the transform applies."""
        return 1

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, tuple[int, int, int, int]]:
        if self.min_crop_threshold <= 0.0:
            return super().get_params_dependent_on_data(params, data)

        height, width = params["shape"][:2]
        limits = {
            "left": self.crop_left,
            "right": self.crop_right,
            "top": self.crop_top,
            "bottom": self.crop_bottom,
        }
        fractions = {side: self.py_random.uniform(0.0, limits[side]) for side in _SIDES}

        # Guarantee the threshold: if no side reached it, force one eligible side to.
        if max(fractions.values()) < self.min_crop_threshold:
            eligible = [side for side in _SIDES if limits[side] >= self.min_crop_threshold]
            chosen = self.py_random.choice(eligible)
            fractions[chosen] = self.py_random.uniform(self.min_crop_threshold, limits[chosen])

        x_min = int(fractions["left"] * width)
        x_max = int((1.0 - fractions["right"]) * width)
        y_min = int(fractions["top"] * height)
        y_max = int((1.0 - fractions["bottom"]) * height)

        # Never emit a degenerate (empty) crop on either axis.
        if x_max <= x_min:
            x_min, x_max = 0, width
        if y_max <= y_min:
            y_min, y_max = 0, height
        return {"crop_coords": (x_min, y_min, x_max, y_max)}
