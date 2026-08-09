"""Every ``Compose`` knob stays reachable from config, without a change here for each."""

from __future__ import annotations

from typing import Any

import albumentations as A
import numpy as np
import pytest

from src.core import Sample
from src.transforms import AlbumentationsTransform


def image() -> np.ndarray:
    return (np.random.default_rng(0).random((16, 16, 3)) * 255).astype(np.uint8)


def sample(**targets: Any) -> Sample:
    return Sample(inputs={"image": image()}, targets=dict(targets))


def flip_and_jitter() -> list[Any]:
    return [A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.5)]


def test_a_seed_makes_the_pipeline_reproducible() -> None:
    """Two runs of the same declaration have to agree, or a sweep means nothing."""
    first = AlbumentationsTransform(flip_and_jitter(), seed=7)(sample())
    second = AlbumentationsTransform(flip_and_jitter(), seed=7)(sample())

    assert np.array_equal(first.inputs["image"], second.inputs["image"])


def test_a_probability_of_zero_leaves_the_sample_alone() -> None:
    original = image()
    given = Sample(inputs={"image": original.copy()}, targets={})

    result = AlbumentationsTransform([A.HorizontalFlip(p=1.0)], p=0.0)(given)

    assert np.array_equal(result.inputs["image"], original)


def test_box_parameters_are_declared_as_a_plain_mapping() -> None:
    """No import path needed: albumentations accepts the mapping YAML already writes."""
    transform = AlbumentationsTransform(
        [A.HorizontalFlip(p=1.0)],
        bbox_params={"coord_format": "yolo", "label_fields": ["classes"]},
    )
    given = Sample(inputs={"image": image()}, targets={})

    augmented = transform._pipeline(image=given.inputs["image"], bboxes=[[0.3, 0.5, 0.2, 0.2]], classes=[1])

    assert augmented["bboxes"][0][0] == pytest.approx(0.7, abs=1e-3)  # mirrored across the frame


def test_telemetry_is_off_by_default_and_can_be_turned_back_on() -> None:
    assert AlbumentationsTransform(flip_and_jitter())._pipeline.telemetry is False
    assert AlbumentationsTransform(flip_and_jitter(), telemetry=True)._pipeline.telemetry is True


def test_the_one_derived_argument_is_refused() -> None:
    """Declaring it would contradict the input names already given."""
    with pytest.raises(ValueError, match="derived here"):
        AlbumentationsTransform(flip_and_jitter(), additional_targets={"right": "image"})


def test_the_operations_keep_the_name_compose_gives_them() -> None:
    """Every argument here is spelled as albumentations spells it, this one included."""
    transform = AlbumentationsTransform(transforms=[A.HorizontalFlip(p=1.0)])

    assert transform(sample()).inputs["image"].shape == (16, 16, 3)


def test_an_unknown_option_fails_where_it_belongs() -> None:
    """Forwarding verbatim means albumentations, not us, decides what it accepts."""
    with pytest.raises(TypeError, match="nonsense"):
        AlbumentationsTransform(flip_and_jitter(), nonsense=1)


def test_the_ordinary_declaration_still_needs_none_of_this() -> None:
    result = AlbumentationsTransform([A.HorizontalFlip(p=1.0)])(sample())

    assert result.inputs["image"].shape == (16, 16, 3)
