"""``AlbumentationsTransform``: one joint augmentation over images and geometric targets."""

from __future__ import annotations

import albumentations as A
import numpy as np
import pytest
import torch
from albumentations.pytorch import ToTensorV2

from src.core import Geometry, Sample
from src.transforms import AlbumentationsTransform

ALWAYS_FLIP = A.HorizontalFlip(p=1.0)


def gradient_image(height: int = 4, width: int = 6) -> np.ndarray:
    """An asymmetric RGB image — flipping it is observable."""
    plane = np.arange(height * width, dtype=np.uint8).reshape(height, width)
    return np.repeat(plane[:, :, None], 3, axis=2)


def gradient_mask(height: int = 4, width: int = 6) -> np.ndarray:
    return np.arange(height * width, dtype=np.int64).reshape(height, width)


def boxed_sample(boxes: list[list[float]], names: list[str], height: int = 100, width: int = 200) -> Sample:
    """One flat picture and its objects, in the pair a BOXES target travels as."""
    return Sample(
        inputs={"image": np.zeros((height, width, 3), dtype=np.uint8)},
        targets={"objects": (np.asarray(boxes, dtype=np.float32).reshape(-1, 4), names)},
    )


def test_applies_the_pipeline_to_the_image() -> None:
    sample = Sample(inputs={"image": gradient_image()}, targets={})

    transformed = AlbumentationsTransform([ALWAYS_FLIP])(sample)

    assert np.array_equal(transformed.inputs["image"], gradient_image()[:, ::-1])


def test_mask_targets_ride_the_same_geometry_as_the_image() -> None:
    """The reason a transform takes a whole Sample: one crop for image and mask alike."""
    sample = Sample(
        inputs={"image": gradient_image()},
        targets={"mask": gradient_mask(), "label": 1},
    )

    transformed = AlbumentationsTransform([A.RandomCrop(height=2, width=3)], targets={"mask": Geometry.MASK})(sample)

    image_plane = transformed.inputs["image"][:, :, 0]
    assert np.array_equal(image_plane, transformed.targets["mask"])
    assert transformed.targets["label"] == 1


def test_a_geometry_arrives_as_its_config_string_too() -> None:
    """Assembly hands members; a hand-written pipeline may spell the same fact as a string."""
    sample = Sample(inputs={"image": gradient_image()}, targets={"mask": gradient_mask()})

    transformed = AlbumentationsTransform([A.RandomCrop(height=2, width=3)], targets={"mask": "mask"})(sample)

    assert transformed.targets["mask"].shape == (2, 3)


def test_an_unknown_geometry_is_refused_naming_the_value_and_the_known_ones() -> None:
    with pytest.raises(ValueError, match="mask.*surface|surface"):
        AlbumentationsTransform([ALWAYS_FLIP], targets={"mask": "surface"})


def test_an_input_that_is_not_pixels_is_refused() -> None:
    """A NONE input would reach albumentations as a label; a BOXES one has no image to be."""
    with pytest.raises(ValueError, match="vector"):
        AlbumentationsTransform([ALWAYS_FLIP], inputs={"image": Geometry.IMAGE, "vector": Geometry.NONE})


def test_several_image_inputs_share_one_sampling() -> None:
    """Two views of a pair must get identical crops — one call, one sampling."""
    sample = Sample(
        inputs={"left": gradient_image(8, 8), "right": gradient_image(8, 8)},
        targets={},
    )

    transformed = AlbumentationsTransform(
        [A.RandomCrop(height=2, width=2)], inputs={"left": Geometry.IMAGE, "right": Geometry.IMAGE}
    )(sample)

    assert np.array_equal(transformed.inputs["left"], transformed.inputs["right"])


def test_undeclared_inputs_pass_through_untouched() -> None:
    embedding = np.ones(4, dtype=np.float32)
    sample = Sample(inputs={"image": gradient_image(), "vector": embedding}, targets={})

    transformed = AlbumentationsTransform([ALWAYS_FLIP])(sample)

    assert np.array_equal(transformed.inputs["vector"], embedding)


def test_the_same_operations_serve_two_schemas_without_leaking_targets() -> None:
    """Ops are shared freely: mask registration belongs to the adapter, not the ops."""
    transforms = [ALWAYS_FLIP]
    AlbumentationsTransform(transforms, targets={"mask": Geometry.MASK})
    without_masks = AlbumentationsTransform(transforms)
    mask = gradient_mask()

    transformed = without_masks(Sample(inputs={"image": gradient_image()}, targets={"mask": mask}))

    assert np.array_equal(transformed.targets["mask"], mask)


def test_a_tensor_ending_pipeline_yields_model_ready_tensors() -> None:
    sample = Sample(inputs={"image": gradient_image()}, targets={"mask": gradient_mask()})

    transformed = AlbumentationsTransform([ToTensorV2()], targets={"mask": Geometry.MASK})(sample)

    assert isinstance(transformed.inputs["image"], torch.Tensor)
    assert transformed.inputs["image"].shape == (3, 4, 6)
    assert isinstance(transformed.targets["mask"], torch.Tensor)


def test_boxes_follow_the_letterbox_geometry_exactly() -> None:
    """Measured on albumentationsx 2.3.7: a 200x100 picture letterboxed to 64x64 scales by
    0.32 and pads 16 rows, so [50, 25, 150, 75] lands on [16, 24, 48, 40]."""
    sample = boxed_sample([[50.0, 25.0, 150.0, 75.0]], ["dog"])

    transformed = AlbumentationsTransform([A.LetterBox(size=(64, 64))], targets={"objects": Geometry.BOXES})(sample)
    boxes, names = transformed.targets["objects"]

    assert np.allclose(boxes, [[16.0, 24.0, 48.0, 40.0]])
    assert names == ["dog"]
    assert transformed.inputs["image"].shape == (64, 64, 3)


def test_a_crop_drops_a_box_and_its_name_together() -> None:
    """The names ride their own field precisely so this pair cannot desynchronise."""
    sample = boxed_sample([[10.0, 10.0, 50.0, 50.0], [150.0, 10.0, 190.0, 50.0]], ["keep", "drop"])

    transformed = AlbumentationsTransform(
        [A.Crop(x_min=0, y_min=0, x_max=60, y_max=100)],
        targets={"objects": Geometry.BOXES},
        min_box_visibility=0.3,
    )(sample)
    boxes, names = transformed.targets["objects"]

    assert boxes.shape == (1, 4)
    assert names == ["keep"]


def test_a_negative_sample_rides_through_as_zero_boxes() -> None:
    """An image with nothing in it is an observation, not a missing value."""
    sample = boxed_sample([], [], height=10, width=10)

    transformed = AlbumentationsTransform([ALWAYS_FLIP], targets={"objects": Geometry.BOXES})(sample)
    boxes, names = transformed.targets["objects"]

    assert boxes.shape == (0, 4)
    assert names == []


def test_boxes_and_a_mask_ride_one_sampling_together() -> None:
    """The multitask case: a detection target and a segmentation target, one crop."""
    sample = Sample(
        inputs={"image": np.zeros((10, 10, 3), dtype=np.uint8)},
        targets={
            "objects": (np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32), ["dog"]),
            "mask": np.arange(100, dtype=np.int64).reshape(10, 10),
        },
    )

    transformed = AlbumentationsTransform([ALWAYS_FLIP], targets={"objects": Geometry.BOXES, "mask": Geometry.MASK})(
        sample
    )

    assert transformed.targets["mask"].shape == (10, 10)
    assert transformed.targets["objects"][0].shape == (1, 4)


def test_two_boxes_targets_are_refused_naming_both() -> None:
    """Measured: albumentationsx 2.3.7 does not plumb label fields through additional targets."""
    with pytest.raises(ValueError, match="defects"):
        AlbumentationsTransform([ALWAYS_FLIP], targets={"furniture": Geometry.BOXES, "defects": Geometry.BOXES})


def test_a_box_knob_without_a_boxes_target_is_refused() -> None:
    """A knob that silently did nothing would read as filtering that never happened."""
    with pytest.raises(ValueError, match="min_box_visibility"):
        AlbumentationsTransform([ALWAYS_FLIP], min_box_visibility=0.3)


def test_the_seam_keeps_bbox_params_to_itself_when_it_carries_boxes() -> None:
    """Two declarations of one fact: the derived one would win silently."""
    with pytest.raises(ValueError, match="bbox_params"):
        AlbumentationsTransform(
            [ALWAYS_FLIP],
            targets={"objects": Geometry.BOXES},
            bbox_params={"coord_format": "yolo"},
        )


def test_a_value_named_after_the_boxes_carrier_is_refused() -> None:
    """``bboxes`` is this seam's own argument; a declared value of that name would collide."""
    with pytest.raises(ValueError, match="bboxes"):
        AlbumentationsTransform([ALWAYS_FLIP], targets={"bboxes": Geometry.MASK})
