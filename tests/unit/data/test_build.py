"""Building the data side from config: schema from tasks, source by suffix, transforms."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pytest

from src.core import Geometry, Sample, Stage
from src.data import ImageLoader, LabelTargetEncoder, MaskTargetEncoder, TableDataModule
from src.data.build import build_transforms
from tests.support.configs import disk_config, disk_data, pipeline_of, resizing_transforms, schema_of
from tests.support.datasets import write_dataset

if TYPE_CHECKING:
    from collections.abc import Callable


class VectorLoader:
    """A non-pixel input for these tests: geometry NONE, resolvable by ``_target_``."""

    geometry: ClassVar[Geometry] = Geometry.NONE

    def __call__(self, value: Any) -> np.ndarray:
        return np.full(4, 1.0, dtype=np.float32)


def plain_vector() -> Callable[[Any], np.ndarray]:
    """A loader that is only a function: no class, so nowhere to declare a geometry."""
    return lambda value: np.full(4, 1.0, dtype=np.float32)


def test_a_loader_declaring_no_geometry_is_treated_as_a_picture_and_the_run_says_so(
    dataset_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The port admits a plain callable, which carries no geometry, and the pipeline then treats the
    column as pixels. Substituting that default is allowed; substituting it silently is not."""
    data = disk_data(dataset_root)
    data["inputs"] = dict(data["inputs"]) | {
        "vector": {"column": "label", "loader": {"_target_": "tests.unit.data.test_build.plain_vector"}}
    }
    with caplog.at_level(logging.INFO, logger="src.data.build"):
        schema = schema_of(disk_config(dataset_root, data=data))

    assert schema.inputs["vector"].geometry is Geometry.IMAGE
    assert "'vector'" in caplog.text and "geometry" in caplog.text


def test_the_schema_takes_targets_from_tasks_and_inputs_from_data(dataset_root: Path) -> None:
    """A target column is declared once, in the task that owns it."""
    schema = schema_of(disk_config(dataset_root))

    assert set(schema.inputs) == {"image"}
    assert set(schema.targets) == {"label"}
    assert isinstance(schema.inputs["image"].loader, ImageLoader)
    assert isinstance(schema.targets["label"].encoder, LabelTargetEncoder)


def test_the_source_format_is_inferred_from_the_suffix(dataset_root: Path) -> None:
    assert isinstance(pipeline_of(disk_config(dataset_root)), TableDataModule)


def test_an_unknown_suffix_lists_the_registered_formats(dataset_root: Path) -> None:
    config = disk_config(dataset_root, data=disk_data(dataset_root, source="a.parquet"))

    with pytest.raises(LookupError, match="csv"):
        pipeline_of(config)


def test_an_explicit_format_overrides_the_suffix(dataset_root: Path) -> None:
    source = {"path": str(dataset_root / "annotations.csv"), "format": "csv"}
    data = disk_data(dataset_root, source=source)

    assert isinstance(pipeline_of(disk_config(dataset_root, data=data)), TableDataModule)


def test_each_source_carries_its_own_format(dataset_root: Path) -> None:
    """A run may combine datasets stored differently; one setting for all could not say that."""
    sources = [
        {"path": str(dataset_root / "annotations.csv"), "format": "csv"},
        str(dataset_root / "annotations.csv"),
    ]
    data = disk_data(dataset_root, source=sources)

    assert isinstance(pipeline_of(disk_config(dataset_root, data=data)), TableDataModule)


def test_setup_returns_the_facts_the_encoders_learned(dataset_root: Path) -> None:
    """The facts are a value handed back, not a bag filled behind the caller's back."""
    facts = pipeline_of(disk_config(dataset_root)).setup()

    assert facts["label"].num_classes == 2


def test_target_geometries_are_derived_from_the_encoders_never_configured(tmp_path: Path) -> None:
    """The transform learns how each target rides geometry from the schema, not from YAML."""
    write_dataset(tmp_path, masks=True)
    config = disk_config(
        tmp_path,
        tasks={
            "mask": {
                "kind": "segmentation",
                "target": "mask",
                "classes": {0: "a", 1: "b", 2: "c"},
                "target_encoder": {"name": "mask", "root": str(tmp_path)},
            }
        },
    )
    schema = schema_of(config)

    transforms = build_transforms(config.transforms, schema)

    assert isinstance(schema.targets["mask"].encoder, MaskTargetEncoder)
    sample = Sample(
        inputs={"image": np.zeros((32, 32, 3), np.uint8)},
        targets={"mask": np.zeros((32, 32), np.int64)},
    )
    transformed = transforms[Stage.TRAIN](sample)
    assert transformed.targets["mask"].shape == (16, 16)  # resized together with the image


def test_no_transforms_section_means_no_transforms(dataset_root: Path) -> None:
    config = disk_config(dataset_root, transforms=None)

    assert build_transforms(config.transforms, schema_of(config)) == {}


@pytest.mark.parametrize(("missing", "taken_from"), [(Stage.TEST, Stage.VAL), (Stage.VAL, Stage.TEST)])
def test_a_missing_eval_pipeline_takes_the_other_eval_stages_and_says_so(
    dataset_root: Path, missing: Stage, taken_from: Stage, caplog: pytest.LogCaptureFixture
) -> None:
    """Evaluation is declared once: the two eval stages complete each other, out loud."""
    declared = {stage: pipeline for stage, pipeline in resizing_transforms().items() if stage != missing}
    config = disk_config(dataset_root, transforms=declared)

    with caplog.at_level(logging.INFO, logger="src.data.build"):
        built = build_transforms(config.transforms, schema_of(config))

    assert built[missing] is built[taken_from]
    assert any(str(missing) in record.message and str(taken_from) in record.message for record in caplog.records)


def test_a_train_pipeline_without_an_eval_pipeline_is_refused_by_name(dataset_root: Path) -> None:
    """A train-only section would evaluate on pictures that were never resized or normalised — in silence."""
    config = disk_config(dataset_root, transforms={"train": resizing_transforms()["train"]})

    with pytest.raises(ValueError, match="val"):
        build_transforms(config.transforms, schema_of(config))


def test_a_sources_own_transforms_are_taken_as_declared(dataset_root: Path) -> None:
    """A source overrides only the stages it names; the rule above is the experiment section's."""
    data = disk_data(dataset_root)
    data["source"] = {"path": data["source"], "transforms": {"train": resizing_transforms(8)["train"]}}
    config = disk_config(dataset_root, data=data)

    module = pipeline_of(config)
    module.setup()

    assert module.dataset(Stage.VAL) is not None  # built as declared: no refusal, nothing completed


def test_a_non_pixel_input_never_enters_the_pipeline_and_reaches_the_batch_untouched(
    dataset_root: Path,
) -> None:
    """The filter the targets side already has, applied to inputs: a NONE-geometry
    column is left out rather than handed to albumentations as a fake picture."""
    data = disk_data(dataset_root)
    data["inputs"] = dict(data["inputs"]) | {
        "embedding": {"column": "label", "loader": {"_target_": "tests.unit.data.test_build.VectorLoader"}}
    }
    config = disk_config(dataset_root, data=data)
    schema = schema_of(config)
    transforms = build_transforms(config.transforms, schema)

    sample = Sample(
        inputs={"image": np.zeros((32, 32, 3), np.uint8), "embedding": np.full(4, 1.0, np.float32)},
        targets={"label": 0},
    )
    transformed = transforms[Stage.TRAIN](sample)

    assert transformed.inputs["embedding"].shape == (4,)  # untouched: never resized, never normalised


@pytest.mark.parametrize(
    ("classes", "why"),
    [({0: "cat", 2: "dog"}, "missing: 1"), ({0: "cat", 1: "cat"}, "duplicated")],
)
def test_a_broken_vocabulary_is_refused_where_its_encoder_is_built_naming_the_task(
    dataset_root: Path, classes: dict[int, str], why: str
) -> None:
    """The index space is the encoder's contract, so the encoder is the one place it is checked — and
    the refusal names the task, which the encoder alone cannot know."""
    tasks = {"label": {"kind": "classification", "target": "label", "classes": classes}}

    with pytest.raises(ValueError, match=rf"Task 'label'.*{why}"):
        schema_of(disk_config(dataset_root, tasks=tasks))
