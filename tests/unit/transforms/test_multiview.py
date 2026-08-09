"""``MultiViewTransform``: one sample input becomes N stacked augmented views."""

from __future__ import annotations

import albumentations as A
import numpy as np
import pytest
import torch
from albumentations.pytorch import ToTensorV2

from src.core import Sample
from src.transforms import AlbumentationsTransform, MultiViewTransform


def make_sample() -> Sample:
    return Sample(
        inputs={"image": torch.zeros(3, 4, 4), "extra": torch.ones(2)},
        targets={"label": torch.tensor(1)},
    )


def test_stacks_n_identical_views_without_a_base_transform() -> None:
    transformed = MultiViewTransform(views=2)(make_sample())

    assert transformed.inputs["image"].shape == (2, 3, 4, 4)


def test_base_transform_runs_per_view_so_views_differ() -> None:
    counter = iter(range(1, 10))

    def brighten(sample: Sample) -> Sample:
        sample.inputs["image"] = sample.inputs["image"] + next(counter)
        return sample

    transformed = MultiViewTransform(views=2, base=brighten)(make_sample())

    views = transformed.inputs["image"]
    assert not torch.equal(views[0], views[1])


def test_other_inputs_and_targets_pass_through() -> None:
    transformed = MultiViewTransform(views=2)(make_sample())

    assert torch.equal(transformed.inputs["extra"], torch.ones(2))
    assert transformed.targets["label"].item() == 1


def test_fewer_than_two_views_is_rejected() -> None:
    with pytest.raises(ValueError, match="two"):
        MultiViewTransform(views=1)


def test_composes_with_an_albumentations_pipeline() -> None:
    """SimCLR-style views: independent sampling is composition, not a flag."""
    sample = Sample(inputs={"image": np.zeros((8, 8, 3), np.uint8)}, targets={})

    transformed = MultiViewTransform(
        views=2,
        base=AlbumentationsTransform([A.RandomCrop(height=4, width=4), A.Normalize(), ToTensorV2()]),
    )(sample)

    views = transformed.inputs["image"]
    assert views.shape == (2, 3, 4, 4)
    assert views.dtype == torch.float32


def test_view_augmentation_never_reaches_the_samples_targets() -> None:
    """Views are stacked, targets are not — a per-view mask would misalign with them."""
    mask = np.arange(64, dtype=np.int64).reshape(8, 8)
    sample = Sample(inputs={"image": np.zeros((8, 8, 3), np.uint8)}, targets={"mask": mask})

    transformed = MultiViewTransform(
        views=2,
        base=AlbumentationsTransform([A.RandomCrop(height=4, width=4)], spatial_targets=["mask"]),
    )(sample)

    assert transformed.targets["mask"].shape == (8, 8)
    assert np.array_equal(transformed.targets["mask"], mask)
