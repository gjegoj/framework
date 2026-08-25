"""``Rotate90``: the picture and its rotation class always agree."""

from __future__ import annotations

import numpy as np

from src.core import Geometry, Sample
from src.transforms import AlbumentationsTransform
from src.transforms.augmentations import QUARTER_TURNS, Rotate90


def cornered() -> np.ndarray:
    """An image whose top-left corner is unlike every other, so a turn is visible."""
    picture = np.zeros((8, 8, 3), dtype=np.uint8)
    picture[0, 0] = 255
    return picture


def turn(seed: int, label: int = 0) -> tuple[np.ndarray, int]:
    transform = AlbumentationsTransform([Rotate90(p=1.0)], label_targets=["angle"], seed=seed)
    result = transform(Sample(inputs={"image": cornered()}, targets={"angle": label}))
    return result.inputs["image"], result.targets["angle"]


def test_the_label_advances_by_the_turns_the_image_took() -> None:
    for seed in range(8):
        picture, angle = turn(seed)
        assert np.array_equal(picture, np.ascontiguousarray(np.rot90(cornered(), angle))), seed


def test_a_label_that_was_not_zero_advances_from_where_it_was() -> None:
    """The label holds the image's current rotation, not the turn just taken."""
    _, from_zero = turn(1, label=0)
    _, from_two = turn(1, label=2)

    assert from_two == (from_zero + 2) % QUARTER_TURNS


def test_a_mask_turns_with_its_image() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0, 0] = 1
    transform = AlbumentationsTransform(
        [Rotate90(p=1.0)], targets={"mask": Geometry.MASK}, label_targets=["angle"], seed=4
    )

    result = transform(Sample(inputs={"image": cornered()}, targets={"mask": mask, "angle": 0}))

    assert np.array_equal(result.targets["mask"], np.ascontiguousarray(np.rot90(mask, result.targets["angle"])))


def test_it_needs_no_column_name_of_its_own() -> None:
    """Binding is the adapter's business; the augmentation only knows the rule."""
    assert "label_key" not in Rotate90(p=1.0).__dict__


def test_every_turn_is_reachable() -> None:
    """A dataset of upright images has to fill all four classes, not two."""
    seen = {turn(seed)[1] for seed in range(40)}

    assert seen == set(range(QUARTER_TURNS))


def test_there_are_four_rotation_classes() -> None:
    assert QUARTER_TURNS == 4
