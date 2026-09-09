"""The ``TargetEncoder`` contract: load before the transforms, encode after; registration; geometry."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.core import Geometry
from src.core.entities import Sample
from src.data.dataset import TableDataset
from src.data.encoders import (
    BoxesTargetEncoder,
    FileTargetEncoder,
    GaussianBinsTargetEncoder,
    LabelTargetEncoder,
    MaskTargetEncoder,
    MultiLabelTargetEncoder,
    ScalarTargetEncoder,
    TargetEncoder,
    VocabularyTargetEncoder,
)
from src.data.registry import target_encoder_registry
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


def test_built_in_encoders_are_registered_for_config() -> None:
    assert set(target_encoder_registry) == {
        "label",
        "multilabel",
        "scalar",
        "mask",
        "boxes",
        "gaussian_bins",
        "linear_bins",
    }
    assert isinstance(target_encoder_registry.create("mask", classes={0: "a", 1: "b"}), MaskTargetEncoder)


def test_encoders_declare_their_geometry() -> None:
    """Geometry is what tells a transform which targets follow the image, and how."""
    assert MaskTargetEncoder(classes={0: "a", 1: "b"}).geometry is Geometry.MASK
    assert LabelTargetEncoder(classes={0: "cat", 1: "dog"}).geometry is Geometry.NONE
    assert ScalarTargetEncoder().geometry is Geometry.NONE


def test_the_encoders_that_read_a_vocabulary_say_so_by_their_base() -> None:
    """``build_target_encoder`` hands ``classes`` to these and refuses it on the rest — by the base, never by a signature."""
    reading: tuple[type[TargetEncoder], ...] = (
        LabelTargetEncoder,
        MultiLabelTargetEncoder,
        BoxesTargetEncoder,
        MaskTargetEncoder,
    )
    not_reading: tuple[type[TargetEncoder], ...] = (ScalarTargetEncoder, GaussianBinsTargetEncoder)

    assert all(issubclass(encoder, VocabularyTargetEncoder) for encoder in reading)
    assert not any(issubclass(encoder, VocabularyTargetEncoder) for encoder in not_reading)


def test_an_encoder_that_reads_files_can_be_told_to_read_through_a_cache(tmp_path: Any) -> None:
    """Only a file-reading encoder has anything to cache; it says so by its base and takes the cache after construction."""
    from collections.abc import Callable, Hashable, Iterable

    from PIL import Image

    from src.data.cache import CacheUsage, LoaderCache

    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(tmp_path / "m.png")

    class Dictionary(LoaderCache):
        """Holds everything it is given, so a second read is a hit — what ``RamCache`` does inside ``warm``."""

        def __init__(self) -> None:
            self.held: dict[Hashable, Any] = {}
            self.reads = 0

        def get(self, key: Hashable) -> Any | None:
            self.reads += 1
            return self.held.get(key)

        def put(self, key: Hashable, value: Any) -> None:
            self.held[key] = value

        def warm(self, keys: Iterable[str], load: Callable[[Any], Any], label: str = "files") -> None:
            raise NotImplementedError

        def usage(self) -> CacheUsage:
            return CacheUsage(files=len(self.held), used_bytes=0, capacity_bytes=0, declined=0, full=False)

        def summarize(self) -> None:
            raise NotImplementedError

    encoder = MaskTargetEncoder(classes={0: "a"}, root=tmp_path)
    assert isinstance(encoder, FileTargetEncoder)
    cache = Dictionary()

    encoder.use_cache(cache)
    first = encoder.load("m.png")
    second = encoder.load("m.png")

    assert list(cache.held) == ["m.png"] and cache.reads == 2  # asked twice, read from disk once
    assert np.array_equal(first, second)


def test_fit_hands_the_fitted_encoder_back() -> None:
    """The fitted encoder is an expression, so ``encoder.fit(column).facts()`` reads in one line."""
    encoder = LabelTargetEncoder(classes={0: "cat", 1: "dog"})

    assert encoder.fit(["cat", "dog"]) is encoder
