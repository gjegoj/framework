"""A tiny image dataset on disk and factories for the objects data tests assemble over and over."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytest
from albumentations.pytorch import ToTensorV2

from src.core import Sample
from src.data import Preprocessor, StandardPreprocessor
from src.data.collate import StackCollator
from src.data.encoders import ImageEncoder, LabelEncoder, MaskEncoder
from src.data.split import Split
from src.data.table import TableDataModule
from src.transforms import AlbumentationsTransform, SampleTransform
from tests.support.declarations import CLASSES

MASK_CLASSES = {0: "background", 1: "pet"}
SIZE = (4, 4)
HALVES = (0.5, 0.5, 0.5)


def pipeline(preprocessor: Preprocessor, size: tuple[int, int] = SIZE) -> SampleTransform:
    """What a stage declares in `configs/transforms`: resize everything, normalize the image, cross into tensors."""
    declared = [A.Resize(*size), A.Normalize(mean=HALVES, std=HALVES), ToTensorV2()]
    return AlbumentationsTransform(declared).with_geometry(**preprocessor.geometries)


@pytest.fixture(scope="session")
def images(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Three 8x6 RGB images and their masks: `a`, `b`, `c` under one root."""
    root = tmp_path_factory.mktemp("images")
    for name, shade in (("a", 30), ("b", 120), ("c", 220)):
        image = np.full((6, 8, 3), shade, dtype=np.uint8)
        image[..., 0] = 255  # red channel saturated: BGR/RGB order becomes observable
        cv2.imwrite(str(root / f"{name}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        mask = np.zeros((6, 8), dtype=np.uint8)
        mask[2:, 3:] = 1
        cv2.imwrite(str(root / f"{name}_mask.png"), mask)
    return root


@pytest.fixture
def image_encoder(images: Path) -> ImageEncoder:
    return ImageEncoder(image_size=SIZE, root=images)


@pytest.fixture
def mask_encoder(images: Path) -> MaskEncoder:
    return MaskEncoder(classes=MASK_CLASSES, root=images)


@pytest.fixture
def label_encoder() -> LabelEncoder:
    return LabelEncoder(classes=CLASSES)


type PreprocessorFactory = Callable[..., StandardPreprocessor]


@pytest.fixture
def make_preprocessor(image_encoder: ImageEncoder, label_encoder: LabelEncoder) -> PreprocessorFactory:
    """A standard preprocessor over the ``image`` input and the ``species`` label; keywords override any part."""

    def make(**overrides: Any) -> StandardPreprocessor:
        parts: dict[str, Any] = {
            "inputs": {"image": image_encoder},
            "targets": {"species": label_encoder},
            "collator": StackCollator(),
        }
        return StandardPreprocessor(**{**parts, **overrides})

    return make


@pytest.fixture
def preprocessor(
    make_preprocessor: PreprocessorFactory, mask_encoder: MaskEncoder, label_encoder: LabelEncoder
) -> StandardPreprocessor:
    return make_preprocessor(targets={"species": label_encoder, "mask": mask_encoder})


@pytest.fixture
def pixels(preprocessor: StandardPreprocessor) -> SampleTransform:
    """The stage pipeline for the standard `preprocessor` fixture."""
    return pipeline(preprocessor)


@pytest.fixture
def row() -> Sample:
    """One annotation row as the table hands it over: paths and names, nothing loaded."""
    return Sample(inputs={"image": "a.png"}, targets={"species": "dog", "mask": "a_mask.png"}, metadata={"row": 0})


@pytest.fixture
def table() -> pd.DataFrame:
    """Twelve rows over the three images, with a label and a number each."""
    return pd.DataFrame(
        {
            "path": [f"{name}.png" for name in "abc" * 4],
            "species": ["cat", "dog", "cat"] * 4,
            "age": [1.0, 2.0, 3.0] * 4,
        }
    )


type ModuleFactory = Callable[..., TableDataModule]


@pytest.fixture
def make_module(table: pd.DataFrame, make_preprocessor: PreprocessorFactory) -> ModuleFactory:
    """A table module over ``table`` divided three ways, each stage running the pixel pipeline; keywords override."""

    def make(**overrides: Any) -> TableDataModule:
        arguments: dict[str, Any] = {
            "source": table,
            "inputs": {"image": "path"},
            "targets": {"species": "species"},
            "preprocessor": make_preprocessor(),
            "split": Split({"train": 0.5, "val": 0.25, "test": 0.25}),
        }
        arguments.update(overrides)
        preprocessor = arguments["preprocessor"]
        arguments.setdefault("transforms", dict.fromkeys(("train", "val", "test"), pipeline(preprocessor)))
        return TableDataModule(**arguments)

    return make


def prepared(module: TableDataModule, splits: Mapping[str, int] | tuple[str, ...]) -> TableDataModule:
    """Set up and fit in one call, for tests that are about what comes after."""
    names = tuple(splits)
    module.setup(names)
    if "train" in names:
        module.fit_preprocessing("train")
    return module
