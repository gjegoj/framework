"""``TableDataModule`` orchestration: source → split → fit encoders → profile → datasets."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from src.core import DataModule, DataProfile, Sample, Stage
from src.data import (
    DataSchema,
    InMemorySource,
    InputColumn,
    LabelTargetEncoder,
    TableDataModule,
    TargetColumn,
    random_split,
)
from tests.support.tables import load_zeros


def make_module(transform_train: bool = False) -> TableDataModule:
    table = pd.DataFrame(
        {
            "path": [f"{index}.jpg" for index in range(8)],
            "label": ["cat", "dog"] * 4,
        }
    )
    schema = DataSchema(
        inputs={"image": InputColumn(column="path", loader=load_zeros)},
        targets={"label": TargetColumn(column="label", encoder=LabelTargetEncoder())},
    )

    def brighten(sample: Sample) -> Sample:
        sample.inputs["image"] = sample.inputs["image"] + 1
        return sample

    return TableDataModule(
        source=InMemorySource(table),
        schema=schema,
        splitter=random_split({Stage.TRAIN: 0.5, Stage.VAL: 0.25, Stage.TEST: 0.25}, seed=42),
        transforms={Stage.TRAIN: brighten} if transform_train else None,
    )


def test_table_data_module_implements_the_data_module_contract() -> None:
    assert isinstance(make_module(), DataModule)


def test_setup_profiles_the_data() -> None:
    module = make_module()
    profile = DataProfile()

    module.setup(profile)

    assert profile.facts("label").num_classes == 2
    assert profile.facts("label").class_names == ["cat", "dog"]


def test_setup_builds_a_dataset_per_stage() -> None:
    module = make_module()
    module.setup(DataProfile())

    sizes = {stage: len(module.dataset(stage)) for stage in Stage}

    assert sizes[Stage.TRAIN] == 4
    assert sum(sizes.values()) == 8


def test_dataset_requires_setup_first() -> None:
    with pytest.raises(RuntimeError, match="setup"):
        make_module().dataset(Stage.TRAIN)


def test_stage_transform_applies_only_to_its_stage() -> None:
    module = make_module(transform_train=True)
    module.setup(DataProfile())

    train_image = module.dataset(Stage.TRAIN)[0].inputs["image"]
    val_image = module.dataset(Stage.VAL)[0].inputs["image"]

    assert torch.equal(train_image, torch.ones(3))
    assert torch.equal(val_image, torch.zeros(3))


def test_datasets_produce_raw_targets_for_transforms_and_collation() -> None:
    module = make_module()
    module.setup(DataProfile())

    sample = module.dataset(Stage.TRAIN)[0]

    assert isinstance(sample.targets["label"], int)


def test_one_source_with_nothing_to_divide_it_is_refused() -> None:
    """A table has to be told how its rows become stages, and there are two ways to say it.

    The refusal lives here rather than in the config section, because it is true of a
    table: a pipeline that reads its own descriptor names its stages there and would be
    forbidden by a rule stated one layer up.
    """
    table = pd.DataFrame({"path": ["0.jpg"], "label": ["cat"]})
    schema = DataSchema(
        inputs={"image": InputColumn(column="path", loader=load_zeros)},
        targets={"label": TargetColumn(column="label", encoder=LabelTargetEncoder())},
    )

    with pytest.raises(ValueError, match="data.split"):
        TableDataModule(source=InMemorySource(table), schema=schema, splitter=None)


def test_per_stage_sources_beside_a_split_are_refused() -> None:
    """Re-cutting a declared partition by fractions would undo the separation it encodes —
    a temporal or per-patient split is decided before the data reaches this framework.
    """
    table = pd.DataFrame({"path": ["0.jpg"], "label": ["cat"]})
    schema = DataSchema(
        inputs={"image": InputColumn(column="path", loader=load_zeros)},
        targets={"label": TargetColumn(column="label", encoder=LabelTargetEncoder())},
    )

    with pytest.raises(ValueError, match="already divided"):
        TableDataModule(
            source={Stage.TRAIN: InMemorySource(table)},
            schema=schema,
            splitter=random_split({Stage.TRAIN: 1.0}, seed=0),
        )
