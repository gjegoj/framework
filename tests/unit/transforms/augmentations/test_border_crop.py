"""``RandomBorderCrop``: a crop worth learning from, and the mark that says it happened."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.core import Sample
from src.transforms import AlbumentationsTransform
from src.transforms.augmentations import RandomBorderCrop


def image() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


def cropped(seed: int, **kwargs: Any) -> Sample:
    transform = AlbumentationsTransform(
        [RandomBorderCrop(crop_left=0.3, crop_right=0.3, crop_top=0.3, crop_bottom=0.3, p=1.0, **kwargs)],
        label_targets=["was_cropped"],
        seed=seed,
    )
    return transform(Sample(inputs={"image": image()}, targets={"was_cropped": 0}))


def test_it_crops() -> None:
    assert cropped(1).inputs["image"].shape[:2] != (100, 100)


def test_it_marks_the_sample_it_cropped() -> None:
    assert cropped(1).targets["was_cropped"] == 1


def test_the_mark_is_not_hard_coded() -> None:
    """A label encoder sorts its vocabulary, so the positive class is not always 1."""
    assert cropped(1, applied_label=0).targets["was_cropped"] == 0


def test_the_mark_may_be_a_class_name() -> None:
    """Encoding runs after the transforms, so the mark is a raw value — a readable
    name beats knowing which index a sorted vocabulary will assign."""
    assert cropped(1, applied_label="cropped").targets["was_cropped"] == "cropped"


def test_a_minimum_crop_is_guaranteed_on_some_side() -> None:
    """Without it a uniform draw may trim two pixels and call the sample cropped."""
    for seed in range(12):
        height, width = cropped(seed, min_crop=0.25).inputs["image"].shape[:2]
        assert min(height, width) <= 76, seed


def test_without_a_minimum_the_parent_decides_alone() -> None:
    shapes = {cropped(seed).inputs["image"].shape[:2] for seed in range(12)}

    assert len(shapes) > 1


def test_the_recorded_crop_matches_the_one_taken() -> None:
    """``applied_config`` feeds albumentations' own reporting; a correction must not desync it."""
    augmentation = RandomBorderCrop(crop_left=0.3, crop_right=0.3, crop_top=0.3, crop_bottom=0.3, min_crop=0.25, p=1.0)
    transform = AlbumentationsTransform([augmentation], label_targets=["was_cropped"], seed=5)

    result = transform(Sample(inputs={"image": image()}, targets={"was_cropped": 0}))

    height, width = result.inputs["image"].shape[:2]
    recorded = augmentation.applied_config
    assert width == pytest.approx(100 * (1 - recorded["crop_left"] - recorded["crop_right"]), abs=2)
    assert height == pytest.approx(100 * (1 - recorded["crop_top"] - recorded["crop_bottom"]), abs=2)


def test_a_minimum_no_side_can_reach_is_refused() -> None:
    with pytest.raises(ValueError, match="min_crop"):
        RandomBorderCrop(crop_left=0.1, crop_right=0.1, crop_top=0.1, crop_bottom=0.1, min_crop=0.5)


def test_a_sample_the_transform_skipped_keeps_its_own_label() -> None:
    """The negative class comes from not applying, so the dataset's labels must be it."""
    transform = AlbumentationsTransform([RandomBorderCrop(p=0.0)], label_targets=["was_cropped"], seed=1)

    result = transform(Sample(inputs={"image": image()}, targets={"was_cropped": 0}))

    assert result.targets["was_cropped"] == 0
