"""Augmentations whose draw is the supervision: what they did to the image becomes its target."""

from __future__ import annotations

from typing import Any

import albumentations as A
import numpy as np
import pytest

from src.core import Geometry, Sample
from src.transforms import AlbumentationsTransform, RandomBorderCrop, Rotate90

TURNS = 4
GEOMETRIES: dict[str, dict[str, Geometry]] = {
    "inputs": {"image": Geometry.IMAGE},
    "targets": {"mask": Geometry.MASK, "angle": Geometry.NONE},
    "auxiliary_inputs": {},
}
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]


def image() -> np.ndarray:
    """A square whose quarters differ, so the turn it took can be read back off it."""
    drawn = np.zeros((4, 4, 3), np.uint8)
    drawn[:2, :2] = 40
    drawn[:2, 2:] = 80
    drawn[2:, :2] = 120
    return drawn


def turned(before: np.ndarray, after: np.ndarray) -> int:
    """How many quarter-turns separate the two images — measured, not assumed."""
    return next(turns for turns in range(TURNS) if np.array_equal(after, np.rot90(before, turns)))


def bound(transform: Any, seed: int, **geometries: Any) -> Any:
    return AlbumentationsTransform([transform], seed=seed).with_geometry(**{**GEOMETRIES, **geometries})


@pytest.mark.parametrize("seed", SEEDS)
def test_the_turn_the_image_took_is_what_its_target_says(seed: int) -> None:
    """The whole point: a folder of upright photographs becomes a balanced four-class task."""
    before = image()

    moved = bound(Rotate90(task="angle"), seed)(Sample(inputs={"image": before}, targets={"angle": 0}))

    assert moved.targets["angle"] == turned(before, np.asarray(moved.inputs["image"]))


@pytest.mark.parametrize("seed", SEEDS)
def test_the_image_and_its_mask_take_the_same_turn(seed: int) -> None:
    before = image()
    mask = np.zeros((4, 4), np.int64)
    mask[:2, :2] = 1

    moved = bound(Rotate90(task="angle"), seed)(Sample(inputs={"image": before}, targets={"mask": mask, "angle": 0}))

    assert turned(mask, np.asarray(moved.targets["mask"])) == moved.targets["angle"]


@pytest.mark.parametrize("seed", SEEDS)
def test_the_target_is_advanced_rather_than_set(seed: int) -> None:
    """It holds the image's current turn, so an already-turned dataset stays truthful."""
    before = image()
    sample = Sample(inputs={"image": before}, targets={"angle": 1})

    moved = bound(Rotate90(task="angle"), seed)(sample)

    assert moved.targets["angle"] == (1 + turned(before, np.asarray(moved.inputs["image"]))) % TURNS


def test_a_turn_nobody_drew_leaves_the_target_alone() -> None:
    moved = bound(Rotate90(task="angle", p=0.0), seed=0)(Sample(inputs={"image": image()}, targets={"angle": 2}))

    assert moved.targets["angle"] == 2


@pytest.mark.parametrize("seed", SEEDS)
def test_the_crop_says_so_in_the_target_it_names(seed: int) -> None:
    cropping = RandomBorderCrop(task="angle", crop_left=0.3, applied_label=1, p=1.0)

    moved = bound(cropping, seed)(Sample(inputs={"image": image()}, targets={"angle": 0}))

    assert moved.targets["angle"] == 1


def test_a_crop_that_did_not_happen_is_the_other_class() -> None:
    """The negative class comes from the augmentation standing down, so the column must already be it."""
    cropping = RandomBorderCrop(task="angle", crop_left=0.3, applied_label=1, p=0.0)

    moved = bound(cropping, seed=0)(Sample(inputs={"image": image()}, targets={"angle": 0}))

    assert moved.targets["angle"] == 0


@pytest.mark.parametrize("seed", SEEDS)
def test_a_crop_too_small_to_learn_from_is_widened_to_the_threshold(seed: int) -> None:
    """A sample trimmed by two pixels teaches nothing while being labelled as cropped."""
    cropping = RandomBorderCrop(task="angle", crop_left=0.3, min_crop=0.25, p=1.0)
    wide = np.zeros((4, 100, 3), np.uint8)

    moved = bound(cropping, seed)(Sample(inputs={"image": wide}, targets={"angle": 0}))

    assert np.asarray(moved.inputs["image"]).shape[1] <= 75


def test_a_threshold_no_side_could_ever_reach_is_refused() -> None:
    with pytest.raises(ValueError, match="min_crop"):
        RandomBorderCrop(task="angle", crop_left=0.1, crop_right=0.1, crop_top=0.1, crop_bottom=0.1, min_crop=0.5)


def test_a_target_no_augmentation_answers_is_left_where_it_is() -> None:
    """The door is opened by the augmentation that answers its task, and by nothing else."""
    moved = bound(A.HorizontalFlip(p=1.0), seed=0)(Sample(inputs={"image": image()}, targets={"angle": 2}))

    assert moved.targets["angle"] == 2


def test_an_augmentation_naming_a_task_the_run_never_declared_is_refused() -> None:
    """Otherwise a typo is an augmentation that quietly writes nothing, for the length of the run."""
    with pytest.raises(ValueError, match="rotation"):
        AlbumentationsTransform([Rotate90(task="rotation")]).with_geometry(**GEOMETRIES)


def test_an_augmentation_answering_a_task_whose_target_is_pixels_is_refused() -> None:
    """A mask moves with the image; an answer is written over. Mixing them up costs a silent epoch."""
    with pytest.raises(ValueError, match="mask"):
        AlbumentationsTransform([Rotate90(task="mask")]).with_geometry(**GEOMETRIES)


def test_two_augmentations_answering_tasks_in_one_pipeline_are_refused() -> None:
    """Measured on albumentationsx 2.3.7: a pipeline routes by kind, so each would rewrite the other's."""
    declared = [Rotate90(task="angle"), RandomBorderCrop(task="cropped")]
    geometries = {**GEOMETRIES, "targets": {"angle": Geometry.NONE, "cropped": Geometry.NONE}}

    with pytest.raises(ValueError, match=r"angle.*cropped"):
        AlbumentationsTransform(declared).with_geometry(**geometries)
