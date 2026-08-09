"""A label declared here is one an augmentation may rewrite; everything else rides along."""

from __future__ import annotations

from typing import Any

import albumentations as A
import numpy as np

from src.core import Sample
from src.transforms import AlbumentationsTransform


class BumpLabel(A.CustomTransformsApplyMixin, A.DualTransform):
    """A stand-in augmentation: leaves the image alone, adds 100 to the bound label."""

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        return img

    def apply_to_label(self, label: Any, **params: Any) -> Any:
        return label + 100


def image() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


def test_a_declared_label_is_rewritten_by_the_augmentation() -> None:
    transform = AlbumentationsTransform([BumpLabel(p=1.0)], label_targets=["angle"])

    result = transform(Sample(inputs={"image": image()}, targets={"angle": 1}))

    assert result.targets["angle"] == 101


def test_an_undeclared_target_rides_through_untouched() -> None:
    """It reaches the pipeline now, and comes back exactly as it went in."""
    transform = AlbumentationsTransform([BumpLabel(p=1.0)], label_targets=["angle"])

    result = transform(Sample(inputs={"image": image()}, targets={"angle": 1, "species": 7}))

    assert result.targets["species"] == 7


def test_an_input_that_is_not_an_image_rides_through_untouched() -> None:
    vector = np.arange(4, dtype=np.float32)

    result = AlbumentationsTransform([A.HorizontalFlip(p=1.0)])(
        Sample(inputs={"image": image(), "vector": vector}, targets={})
    )

    assert np.array_equal(result.inputs["vector"], vector)


def test_declaring_no_label_leaves_the_pipeline_as_it_was() -> None:
    """The augmentation has nothing bound to it, so the label is not its business."""
    transform = AlbumentationsTransform([BumpLabel(p=1.0)])

    result = transform(Sample(inputs={"image": image()}, targets={"angle": 1}))

    assert result.targets["angle"] == 1


def test_images_and_masks_still_share_one_geometry() -> None:
    """The change must not disturb what the adapter already guaranteed."""
    picture = np.zeros((8, 8, 3), dtype=np.uint8)
    picture[:, :4] = 255
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:, :4] = 1
    transform = AlbumentationsTransform([A.HorizontalFlip(p=1.0)], spatial_targets=["mask"])

    result = transform(Sample(inputs={"image": picture}, targets={"mask": mask}))

    assert result.inputs["image"][0, 0, 0] == 0
    assert result.targets["mask"][0, 0] == 0
