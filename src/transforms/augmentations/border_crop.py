"""A border crop that is always worth learning from, and says when it happened."""

from __future__ import annotations

from typing import Any, override

import albumentations as A

SIDES = ("left", "right", "top", "bottom")
"""The borders a crop may trim, named once so a limit, a floor and a draw all spell them alike."""


class RandomBorderCrop(A.CustomTransformsApplyMixin, A.RandomCropFromBorders):
    """Trim a random strip from each border, and mark the sample as trimmed.

    ``min_crop`` is why this exists rather than its parent: a uniform draw may take two pixels, and a
    sample cropped by two pixels teaches nothing while being labelled as cropped. One side is held to
    the threshold, so every sample the positive class is written on carries a crop worth seeing.

    The negative class comes from the augmentation standing down, which is what ``p`` is for: the
    target must already hold that class throughout, and this writes the other one over it.

    Parameters:
        task: The task this answers; its target says whether the picture was trimmed.
        crop_left: Largest fraction of the width taken from the left; the rest read alike.
        crop_right: Largest fraction of the width taken from the right.
        crop_top: Largest fraction of the height taken from the top.
        crop_bottom: Largest fraction of the height taken from the bottom.
        min_crop: Fraction at least one side reaches; ``0`` asks for no guarantee.
        applied_label: What the target says once trimmed. Encoding runs after the transforms, so
            write the class as the table writes it rather than the index a vocabulary gives it.
        p: How often it trims at all — the share of samples that carry the positive class.
    """

    def __init__(
        self,
        task: str,
        *,
        crop_left: float = 0.1,
        crop_right: float = 0.1,
        crop_top: float = 0.1,
        crop_bottom: float = 0.1,
        min_crop: float = 0.0,
        applied_label: int | str = 1,
        p: float = 1.0,
    ) -> None:
        super().__init__(crop_left=crop_left, crop_right=crop_right, crop_top=crop_top, crop_bottom=crop_bottom, p=p)
        largest = max(crop_left, crop_right, crop_top, crop_bottom)
        if min_crop > largest:
            raise ValueError(
                f"min_crop {min_crop} is above every per-side maximum (the largest is {largest}), so no "
                f"side could ever reach it and no crop could ever be drawn."
            )
        self.task = task
        self.min_crop = min_crop
        self.applied_label = applied_label

    def apply_to_label(self, label: Any, **params: Any) -> int | str:
        return self.applied_label

    @override
    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, tuple[int, int, int, int]]:
        height, width = params["shape"][:2]
        taken = self._shares()
        left, top = int(taken["left"] * width), int(taken["top"] * height)
        right, bottom = width - int(taken["right"] * width), height - int(taken["bottom"] * height)
        # A pixel survives on each axis whatever was drawn, as the parent's own draw also guarantees.
        return {"crop_coords": (left, top, max(left + 1, right), max(top + 1, bottom))}

    def _shares(self) -> dict[str, float]:
        """The fraction taken from each side, with one side held to ``min_crop`` where one is asked.

        Held in the draw rather than corrected after it: a crop redrawn until it is large enough has
        no bound on how long that takes, and one widened afterwards is widened on the side that was
        already smallest, which is the opposite of a uniform choice.
        """
        limits = {side: float(getattr(self, f"crop_{side}")) for side in SIDES}
        floors = dict.fromkeys(SIDES, 0.0)
        if self.min_crop > 0.0:
            floors[self.py_random.choice([side for side, limit in limits.items() if limit >= self.min_crop])] = (
                self.min_crop
            )
        return {side: self.py_random.uniform(floors[side], limits[side]) for side in SIDES}
