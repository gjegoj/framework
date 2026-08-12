"""Every column loads before the transforms; target columns encode after.

``load`` is one table cell into the form the transforms see — identity for a
value, a file read for a mask, because geometry needs pixels to move. ``encode``
is the post-transform value into the training form, which is what lets an
augmentation write a raw class name or a plain number and have the encoder make
training sense of it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.core.entities import Sample
from src.data.dataset import TableDataset
from src.data.encoders import GaussianBinsTargetEncoder, LabelTargetEncoder, TargetEncoder
from src.data.schema import DataSchema, InputColumn, TargetColumn


class RecordingTransform:
    """Remembers the targets it was shown, and may rewrite one — a stand-in augmentation."""

    def __init__(self, write: dict[str, Any] | None = None) -> None:
        self.saw: dict[str, Any] | None = None
        self._write = write or {}

    def __call__(self, sample: Sample) -> Sample:
        self.saw = dict(sample.targets)
        sample.targets.update(self._write)
        return sample


class MaskStyleEncoder(TargetEncoder):
    """Does its work in ``load``, as the real mask encoder does: cell in, pixels out."""

    spatial = True

    def load(self, value: Any) -> np.ndarray:
        return np.full((4, 4), 7, np.uint8)

    def encode(self, value: Any) -> Any:
        return value


def table() -> pd.DataFrame:
    return pd.DataFrame({"pixels": [0], "label": ["intact"], "warmth": [0.0]})


def identity_input() -> dict[str, InputColumn]:
    return {"image": InputColumn(column="pixels", loader=lambda value: np.zeros((4, 4, 3), np.uint8))}


def fitted_label_encoder() -> LabelTargetEncoder:
    encoder = LabelTargetEncoder(classes={0: "intact", 1: "cropped"})
    encoder.fit(["intact"])
    return encoder


def test_a_value_target_reaches_the_transform_raw_and_leaves_encoded() -> None:
    transform = RecordingTransform()
    schema = DataSchema(
        inputs=identity_input(),
        targets={"label": TargetColumn(column="label", encoder=fitted_label_encoder())},
    )

    sample = TableDataset(table(), schema, transform)[0]

    assert transform.saw is not None
    assert transform.saw["label"] == "intact"  # raw at transform time: load is identity
    assert sample.targets["label"] == 0  # encoded on the way out


def test_an_augmentations_raw_output_is_encoded_by_the_declared_vocabulary() -> None:
    """The border-crop idiom: the augmentation writes a class NAME, the encoder finds its index."""
    transform = RecordingTransform(write={"label": "cropped"})
    schema = DataSchema(
        inputs=identity_input(),
        targets={"label": TargetColumn(column="label", encoder=fitted_label_encoder())},
    )

    assert TableDataset(table(), schema, transform)[0].targets["label"] == 1


def test_an_online_scalar_is_encoded_into_bins_after_the_transform() -> None:
    """The reason this plan exists: gaussian_bins over a target an augmentation generates."""
    encoder = GaussianBinsTargetEncoder(bins=8, low=3000, high=4600)
    transform = RecordingTransform(write={"warmth": 3800.0})
    schema = DataSchema(
        inputs=identity_input(),
        targets={"warmth": TargetColumn(column="warmth", encoder=encoder)},
    )

    encoded = TableDataset(table(), schema, transform)[0].targets["warmth"]

    assert np.shape(encoded) == (8,)
    assert np.isclose(np.sum(encoded), 1.0)


def test_without_a_transform_targets_still_arrive_encoded() -> None:
    """The val/test path: no transform declared, encoding must not be skipped."""
    schema = DataSchema(
        inputs=identity_input(),
        targets={"label": TargetColumn(column="label", encoder=fitted_label_encoder())},
    )

    assert TableDataset(table(), schema, transform=None)[0].targets["label"] == 0


def test_a_mask_style_target_reaches_the_transform_as_pixels() -> None:
    """A mask encoder loads its file before the pipeline, or geometry has nothing to follow."""
    transform = RecordingTransform()
    schema = DataSchema(
        inputs=identity_input(),
        targets={"mask": TargetColumn(column="label", encoder=MaskStyleEncoder())},
    )

    sample = TableDataset(table(), schema, transform)[0]

    assert transform.saw is not None
    assert isinstance(transform.saw["mask"], np.ndarray)  # pixels at transform time
    assert np.array_equal(sample.targets["mask"], transform.saw["mask"])  # identity encode after


def test_a_target_columns_loader_is_its_encoders_pre_transform_half() -> None:
    """One call shape for every kind of column: the dataset and the cache warm both
    say ``column.loader(cell)``, whatever the column is."""
    value_column = TargetColumn(column="label", encoder=fitted_label_encoder())
    mask_column = TargetColumn(column="label", encoder=MaskStyleEncoder())

    assert value_column.loader("intact") == "intact"  # identity for a value encoder
    assert isinstance(mask_column.loader("whatever.png"), np.ndarray)  # the mask encoder's read
