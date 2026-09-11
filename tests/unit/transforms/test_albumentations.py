"""One pipeline call moves everything pixel-bound; what moves is derived from the encoders, never declared."""

from __future__ import annotations

import pickle
from typing import Any

import albumentations as A
import numpy as np
import pytest
import torch
from albumentations.pytorch import ToTensorV2
from torch import Tensor

from src.core import Geometry, Sample
from src.transforms import AlbumentationsTransform, GeometryAware

GEOMETRIES = {"inputs": {"image": Geometry.IMAGE}, "targets": {"mask": Geometry.MASK}, "auxiliary_inputs": {}}
HALVES = (0.5, 0.5, 0.5)


@pytest.fixture
def sample() -> Sample:
    """A 6x8 picture whose right half is white, with the matching mask and a label nobody should touch."""
    picture = np.zeros((6, 8, 3), np.uint8)
    picture[:, 4:] = 255
    mask = np.zeros((6, 8), np.int64)
    mask[:, 4:] = 1
    return Sample(inputs={"image": picture}, targets={"mask": mask, "label": "cat"}, metadata={"row": 3})


def flip() -> AlbumentationsTransform:
    return AlbumentationsTransform([A.HorizontalFlip(p=1.0)])


def test_is_geometry_aware_and_binds_to_a_picklable_pipeline() -> None:
    bound = flip().with_geometry(**GEOMETRIES)

    assert isinstance(flip(), GeometryAware)
    assert callable(pickle.loads(pickle.dumps(bound)))


def test_moves_the_picture_and_its_mask_together_and_leaves_the_rest(sample: Sample) -> None:
    moved = flip().with_geometry(**GEOMETRIES)(sample)

    assert np.asarray(moved.inputs["image"])[0, 0, 0] == 255 and np.asarray(moved.targets["mask"])[0, 0] == 1
    assert moved.targets["label"] == "cat" and moved.metadata == {"row": 3}


def test_the_picture_may_carry_any_name(sample: Sample) -> None:
    renamed = Sample(inputs={"photo": sample.inputs["image"]}, targets=sample.targets)

    moved = flip().with_geometry(**{**GEOMETRIES, "inputs": {"photo": Geometry.IMAGE}})(renamed)

    assert np.asarray(moved.inputs["photo"])[0, 0, 0] == 255 and np.asarray(moved.targets["mask"])[0, 0] == 1


def test_a_value_the_sample_lacks_is_simply_not_moved(sample: Sample) -> None:
    unlabeled = Sample(inputs=sample.inputs)

    moved = flip().with_geometry(**GEOMETRIES)(unlabeled)

    assert np.asarray(moved.inputs["image"])[0, 0, 0] == 255 and moved.targets == {}


def test_a_stage_pipeline_normalizes_the_picture_alone_and_crosses_both_into_tensors(sample: Sample) -> None:
    """What every `configs/transforms` group declares: the geometry decides what each value is put through."""
    declared = [A.Resize(3, 4), A.Normalize(mean=HALVES, std=HALVES), ToTensorV2()]

    prepared = AlbumentationsTransform(declared).with_geometry(**GEOMETRIES)(sample)

    image, mask = prepared.inputs["image"], prepared.targets["mask"]
    assert isinstance(image, Tensor) and image.shape == (3, 3, 4) and image.dtype is torch.float32
    assert image[:, :, 0].eq(-1.0).all() and image[:, :, -1].eq(1.0).all()  # (pixel / 255 - 0.5) / 0.5
    assert isinstance(mask, Tensor) and mask.shape == (3, 4) and set(mask.unique().tolist()) == {0, 1}


@pytest.mark.parametrize(
    ("declaration", "geometries", "reason"),
    [
        pytest.param({"additional_targets": {}}, GEOMETRIES, "derived", id="additional_targets by hand"),
        pytest.param({}, {**GEOMETRIES, "inputs": {"text": Geometry.NONE}}, "pixels", id="a non-pixel input"),
        pytest.param({}, {**GEOMETRIES, "inputs": {}}, "image input", id="no image at all"),
        pytest.param({}, {**GEOMETRIES, "targets": {"image": Geometry.MASK}}, "more than one role", id="two roles"),
        pytest.param({}, {**GEOMETRIES, "targets": {"objects": Geometry.BOXES}}, "pixels", id="a boxes target"),
    ],
)
def test_refuses_a_binding_that_could_silently_misroute_a_value(
    declaration: dict[str, Any], geometries: dict[str, dict[str, Geometry]], reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        AlbumentationsTransform([A.Resize(3, 4)], **declaration).with_geometry(**geometries)
