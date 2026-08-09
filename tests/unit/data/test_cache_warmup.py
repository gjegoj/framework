"""Warming happens once, in the parent process, over the stages that repeat."""

from __future__ import annotations

from collections.abc import Sized
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pandas as pd

from src.core import DataProfile, Sample, Stage
from src.core.ports import SampleTransform
from src.data import (
    DataSchema,
    ImageLoader,
    InMemorySource,
    InputColumn,
    LabelTargetEncoder,
    RamCache,
    TableDataModule,
    TargetColumn,
    cached,
    random_split,
)

FRACTIONS = {Stage.TRAIN: 0.5, Stage.VAL: 0.25, Stage.TEST: 0.25}


def dataset(root: Path, rows: int = 8) -> pd.DataFrame:
    root.mkdir(parents=True, exist_ok=True)
    for index in range(rows):
        cv2.imwrite(str(root / f"{index}.png"), np.full((8, 8, 3), index, dtype=np.uint8))
    return pd.DataFrame({"path": [f"{index}.png" for index in range(rows)], "label": ["cat", "dog"] * (rows // 2)})


def module(
    root: Path,
    cache: RamCache | None,
    transforms: dict[Stage, SampleTransform] | None = None,
) -> TableDataModule:
    loader = ImageLoader(root=root)
    schema = DataSchema(
        inputs={"image": InputColumn(column="path", loader=cached(loader, cache) if cache else loader)},
        targets={"label": TargetColumn(column="label", encoder=LabelTargetEncoder())},
    )
    return TableDataModule(
        source=InMemorySource(dataset(root)),
        schema=schema,
        splitter=random_split(FRACTIONS, seed=42),
        transforms=transforms,
        cache=cache,
    )


def held(cache: RamCache, rows: int = 8) -> int:
    return sum(cache.get(f"{index}.png") is not None for index in range(rows))


def test_train_and_val_files_are_in_memory_after_setup(tmp_path: Path) -> None:
    cache = RamCache(max_gib=1.0)

    module(tmp_path, cache).setup(DataProfile())

    assert held(cache) == 6


def test_the_test_stage_is_not_cached(tmp_path: Path) -> None:
    """It is read once; RAM spent on it buys nothing."""
    cache = RamCache(max_gib=1.0)

    module(tmp_path, cache).setup(DataProfile())

    assert held(cache) < 8


def test_a_module_without_a_cache_behaves_as_before(tmp_path: Path) -> None:
    built = module(tmp_path, None)

    built.setup(DataProfile())

    assert len(cast("Sized", built.dataset(Stage.TRAIN))) == 4


def test_samples_are_identical_with_and_without_a_cache(tmp_path: Path) -> None:
    """A cache is an optimisation; it must not change a single pixel."""
    plain = module(tmp_path, None)
    warmed = module(tmp_path, RamCache(max_gib=1.0))
    plain.setup(DataProfile())
    warmed.setup(DataProfile())

    for index in range(len(cast("Sized", plain.dataset(Stage.TRAIN)))):
        expected = plain.dataset(Stage.TRAIN)[index].inputs["image"]
        assert np.array_equal(warmed.dataset(Stage.TRAIN)[index].inputs["image"], expected)


def test_augmentation_still_varies_though_the_read_does_not(tmp_path: Path) -> None:
    """Only the decoded file is cached; what happens to it afterwards stays random."""
    rng = np.random.default_rng(0)

    def jitter(sample: Sample) -> Sample:
        sample.inputs["image"] = sample.inputs["image"] + int(rng.integers(1, 100))
        return sample

    built = module(tmp_path, RamCache(max_gib=1.0), transforms={Stage.TRAIN: jitter})
    built.setup(DataProfile())

    first = built.dataset(Stage.TRAIN)[0].inputs["image"]
    second = built.dataset(Stage.TRAIN)[0].inputs["image"]

    assert not np.array_equal(first, second)


def test_a_cached_read_is_not_corrupted_by_the_transform_that_follows(tmp_path: Path) -> None:
    """The transform mutates the sample it is given; the cached array must survive it."""

    def brighten(sample: Sample) -> Sample:
        sample.inputs["image"] = sample.inputs["image"] + 100
        return sample

    cache = RamCache(max_gib=1.0)
    built = module(tmp_path, cache, transforms={Stage.TRAIN: brighten})
    built.setup(DataProfile())
    before: dict[str, np.ndarray] = {}
    for index in range(8):
        key = f"{index}.png"
        if (stored := cache.get(key)) is not None:
            before[key] = stored.copy()

    for index in range(len(cast("Sized", built.dataset(Stage.TRAIN)))):
        built.dataset(Stage.TRAIN)[index]

    for key, untouched in before.items():
        current = cache.get(key)
        assert current is not None, key
        assert np.array_equal(current, untouched), key
