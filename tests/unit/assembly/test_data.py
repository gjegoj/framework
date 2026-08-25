"""Building the data side from config: schema from tasks, source by suffix, transforms."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from src.assembly.data import build_data_module, build_data_schema, build_transforms
from src.core import DataProfile, Geometry, Sample, Stage
from src.data import ImageLoader, LabelTargetEncoder, MaskTargetEncoder, TableDataModule
from tests.support.configs import disk_config, disk_data
from tests.support.datasets import write_dataset


class VectorLoader:
    """A non-pixel input for these tests: geometry NONE, resolvable by ``_target_``."""

    geometry: ClassVar[Geometry] = Geometry.NONE

    def __call__(self, value: Any) -> np.ndarray:
        return np.full(4, 1.0, dtype=np.float32)


def test_the_schema_takes_targets_from_tasks_and_inputs_from_data(dataset_root: Path) -> None:
    """A target column is declared once, in the task that owns it."""
    schema = build_data_schema(disk_config(dataset_root))

    assert set(schema.inputs) == {"image"}
    assert set(schema.targets) == {"label"}
    assert isinstance(schema.inputs["image"].loader, ImageLoader)
    assert isinstance(schema.targets["label"].encoder, LabelTargetEncoder)


def test_the_source_format_is_inferred_from_the_suffix(dataset_root: Path) -> None:
    assert isinstance(build_data_module(disk_config(dataset_root)), TableDataModule)


def test_an_unknown_suffix_lists_the_registered_formats(dataset_root: Path) -> None:
    config = disk_config(dataset_root, data=disk_data(dataset_root, source="a.parquet"))

    with pytest.raises(LookupError, match="csv"):
        build_data_module(config)


def test_an_explicit_format_overrides_the_suffix(dataset_root: Path) -> None:
    source = {"path": str(dataset_root / "annotations.csv"), "format": "csv"}
    data = disk_data(dataset_root, source=source)

    assert isinstance(build_data_module(disk_config(dataset_root, data=data)), TableDataModule)


def test_each_source_carries_its_own_format(dataset_root: Path) -> None:
    """A run may combine datasets stored differently; one setting for all could not say that."""
    sources = [
        {"path": str(dataset_root / "annotations.csv"), "format": "csv"},
        str(dataset_root / "annotations.csv"),
    ]
    data = disk_data(dataset_root, source=sources)

    assert isinstance(build_data_module(disk_config(dataset_root, data=data)), TableDataModule)


def test_the_assembled_module_profiles_its_data(dataset_root: Path) -> None:
    profile = DataProfile()

    build_data_module(disk_config(dataset_root)).setup(profile)

    assert profile.facts("label").num_classes == 2


def test_target_geometries_are_derived_from_the_encoders_never_configured(tmp_path: Path) -> None:
    """The transform learns how each target rides geometry from the schema, not from YAML."""
    write_dataset(tmp_path, masks=True)
    config = disk_config(
        tmp_path,
        tasks={
            "mask": {
                "preset": "segmentation",
                "target": "mask",
                "target_encoder": {"name": "mask", "num_classes": 3, "root": str(tmp_path)},
            }
        },
    )
    schema = build_data_schema(config)

    transforms = build_transforms(config, schema)

    assert isinstance(schema.targets["mask"].encoder, MaskTargetEncoder)
    sample = Sample(
        inputs={"image": np.zeros((32, 32, 3), np.uint8)},
        targets={"mask": np.zeros((32, 32), np.int64)},
    )
    transformed = transforms[Stage.TRAIN](sample)
    assert transformed.targets["mask"].shape == (16, 16)  # resized together with the image


def test_no_transforms_section_means_no_transforms(dataset_root: Path) -> None:
    config = disk_config(dataset_root, transforms=None)

    assert build_transforms(config, build_data_schema(config)) == {}


def test_a_table_still_needs_an_input_column(dataset_root: Path) -> None:
    """The rule holds where the table's schema is built, which is the only place it is true.

    Stated in the config section instead, it also forbade the pipelines that have no
    columns at all — a vendor dataset reads its images from its own descriptor.
    """
    config = disk_config(dataset_root, data={"source": "a.csv", "inputs": {}})

    with pytest.raises(ValueError, match="data.inputs"):
        build_data_schema(config)


def test_a_section_with_no_columns_is_a_valid_declaration(dataset_root: Path) -> None:
    """A vendor pipeline names its images in its own descriptor, so an empty `inputs` is a
    declaration rather than a mistake — and refusing it in the schema would forbid a family.
    """
    config = disk_config(dataset_root, data={"source": "coco8.yaml", "inputs": {}})

    assert config.data.inputs == {}


def test_a_non_pixel_input_never_enters_the_pipeline_and_reaches_the_batch_untouched(
    dataset_root: Path,
) -> None:
    """The filter the targets side already has, applied to inputs: a NONE-geometry
    column is left out rather than handed to albumentations as a fake picture."""
    data = disk_data(dataset_root)
    data["inputs"] = dict(data["inputs"]) | {
        "embedding": {"column": "label", "loader": {"_target_": "tests.unit.assembly.test_data.VectorLoader"}}
    }
    config = disk_config(dataset_root, data=data)
    schema = build_data_schema(config)
    transforms = build_transforms(config, schema)

    sample = Sample(
        inputs={"image": np.zeros((32, 32, 3), np.uint8), "embedding": np.full(4, 1.0, np.float32)},
        targets={"label": 0},
    )
    transformed = transforms[Stage.TRAIN](sample)

    assert transformed.inputs["embedding"].shape == (4,)  # untouched: never resized, never normalised
