"""Auxiliary inputs inside the albumentations pipeline: mask geometry, no normalisation."""

from __future__ import annotations

import albumentations as A
import numpy as np
import pytest

from src.core import Geometry, Sample
from src.transforms import AlbumentationsTransform


def image() -> np.ndarray:
    picture = np.zeros((8, 8, 3), dtype=np.uint8)
    picture[:, :4] = 255
    return picture


def half_mask() -> np.ndarray:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:, :4] = 1
    return mask


def test_an_auxiliary_input_follows_the_images_geometry() -> None:
    """A crop taken from the image is the same crop taken from the mask that bounds it."""
    transform = AlbumentationsTransform([A.HorizontalFlip(p=1.0)], auxiliary_inputs={"lesion": Geometry.MASK})

    result = transform(Sample(inputs={"image": image()}, targets={}, auxiliary_inputs={"lesion": half_mask()}))

    assert result.inputs["image"][0, 0, 0] == 0  # flipped
    assert np.asarray(result.auxiliary_inputs["lesion"]).squeeze()[0, 0] == 0  # flipped with it


def test_normalize_leaves_an_auxiliary_input_untouched() -> None:
    """Registered as mask-kind, so the pixel statistics never touch it. Measured as an
    image target instead: a 0/1 mask becomes floats -2.118..-1.787, and a ``> 0`` region
    test then finds nothing at all."""
    transform = AlbumentationsTransform(
        [A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))], auxiliary_inputs={"lesion": Geometry.MASK}
    )

    result = transform(Sample(inputs={"image": image()}, targets={}, auxiliary_inputs={"lesion": half_mask()}))

    assert np.array_equal(np.asarray(result.auxiliary_inputs["lesion"]).squeeze(), half_mask())


def test_an_auxiliary_input_is_written_back_for_the_next_transform_in_a_chain() -> None:
    """Whatever the pipeline did to it is what a later transform must see."""
    transform = AlbumentationsTransform([A.Resize(4, 4)], auxiliary_inputs={"lesion": Geometry.MASK})

    result = transform(Sample(inputs={"image": image()}, targets={}, auxiliary_inputs={"lesion": half_mask()}))

    assert np.asarray(result.auxiliary_inputs["lesion"]).squeeze().shape == (4, 4)


def test_a_name_shared_across_pipeline_roles_is_refused() -> None:
    """Every declared name becomes a kwarg of one pipeline call; a duplicate is a silent
    overwrite, so it is refused at construction with the roles named."""
    with pytest.raises(ValueError, match="lesion"):
        AlbumentationsTransform(
            [A.HorizontalFlip(p=1.0)],
            targets={"lesion": Geometry.MASK},
            auxiliary_inputs={"lesion": Geometry.MASK},
        )


def test_an_undeclared_auxiliary_input_rides_through_untouched() -> None:
    """Only declared keys reach the pipeline — the same rule inputs and targets follow."""
    transform = AlbumentationsTransform([A.HorizontalFlip(p=1.0)])
    carried = half_mask()

    result = transform(Sample(inputs={"image": image()}, targets={}, auxiliary_inputs={"lesion": carried}))

    assert np.array_equal(result.auxiliary_inputs["lesion"], carried)


def test_a_mask_input_gets_mask_treatment_and_stays_a_model_input() -> None:
    """The conditioned-model case: image *and* mask into the model. It rides the
    pipeline as a mask — measured as an image target instead, a ~30 degree rotation
    smears a binary edge into 11 grey levels and Normalize rewrites every value."""
    transform = AlbumentationsTransform(
        [A.HorizontalFlip(p=1.0), A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))],
        inputs={"image": Geometry.IMAGE, "lesion_mask": Geometry.MASK},
    )

    result = transform(Sample(inputs={"image": image(), "lesion_mask": half_mask()}, targets={}))

    flipped = np.asarray(result.inputs["lesion_mask"]).squeeze()
    assert flipped[0, 0] == 0  # flipped with the image
    assert set(np.unique(flipped).tolist()) == {0, 1}  # Normalize never touched it


def test_a_mask_input_shares_the_name_check_with_every_other_role() -> None:
    with pytest.raises(ValueError, match="lesion"):
        AlbumentationsTransform(
            [A.HorizontalFlip(p=1.0)],
            inputs={"image": Geometry.IMAGE, "lesion": Geometry.MASK},
            auxiliary_inputs={"lesion": Geometry.MASK},
        )
