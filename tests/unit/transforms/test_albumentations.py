"""``AlbumentationsTransform``: one joint augmentation over images and spatial targets."""

from __future__ import annotations

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from src.core import Sample
from src.transforms import AlbumentationsTransform

ALWAYS_FLIP = A.HorizontalFlip(p=1.0)


def gradient_image(height: int = 4, width: int = 6) -> np.ndarray:
    """An asymmetric RGB image — flipping it is observable."""
    plane = np.arange(height * width, dtype=np.uint8).reshape(height, width)
    return np.repeat(plane[:, :, None], 3, axis=2)


def gradient_mask(height: int = 4, width: int = 6) -> np.ndarray:
    return np.arange(height * width, dtype=np.int64).reshape(height, width)


def test_applies_the_pipeline_to_the_image() -> None:
    sample = Sample(inputs={"image": gradient_image()}, targets={})

    transformed = AlbumentationsTransform([ALWAYS_FLIP])(sample)

    assert np.array_equal(transformed.inputs["image"], gradient_image()[:, ::-1])


def test_spatial_targets_ride_the_same_geometry_as_the_image() -> None:
    """The reason a transform takes a whole Sample: one crop for image and mask alike."""
    sample = Sample(
        inputs={"image": gradient_image()},
        targets={"mask": gradient_mask(), "label": 1},
    )

    transformed = AlbumentationsTransform([A.RandomCrop(height=2, width=3)], spatial_targets=["mask"])(sample)

    image_plane = transformed.inputs["image"][:, :, 0]
    assert np.array_equal(image_plane, transformed.targets["mask"])
    assert transformed.targets["label"] == 1


def test_several_image_inputs_share_one_sampling() -> None:
    """Two views of a pair must get identical crops — one call, one sampling."""
    sample = Sample(
        inputs={"left": gradient_image(8, 8), "right": gradient_image(8, 8)},
        targets={},
    )

    transformed = AlbumentationsTransform([A.RandomCrop(height=2, width=2)], image_inputs=["left", "right"])(sample)

    assert np.array_equal(transformed.inputs["left"], transformed.inputs["right"])


def test_undeclared_inputs_pass_through_untouched() -> None:
    embedding = np.ones(4, dtype=np.float32)
    sample = Sample(inputs={"image": gradient_image(), "vector": embedding}, targets={})

    transformed = AlbumentationsTransform([ALWAYS_FLIP])(sample)

    assert np.array_equal(transformed.inputs["vector"], embedding)


def test_the_same_operations_serve_two_schemas_without_leaking_targets() -> None:
    """Ops are shared freely: mask registration belongs to the adapter, not the ops."""
    transforms = [ALWAYS_FLIP]
    AlbumentationsTransform(transforms, spatial_targets=["mask"])
    without_masks = AlbumentationsTransform(transforms)
    mask = gradient_mask()

    transformed = without_masks(Sample(inputs={"image": gradient_image()}, targets={"mask": mask}))

    assert np.array_equal(transformed.targets["mask"], mask)


def test_a_tensor_ending_pipeline_yields_model_ready_tensors() -> None:
    sample = Sample(inputs={"image": gradient_image()}, targets={"mask": gradient_mask()})

    transformed = AlbumentationsTransform([ToTensorV2()], spatial_targets=["mask"])(sample)

    assert isinstance(transformed.inputs["image"], torch.Tensor)
    assert transformed.inputs["image"].shape == (3, 4, 6)
    assert isinstance(transformed.targets["mask"], torch.Tensor)
