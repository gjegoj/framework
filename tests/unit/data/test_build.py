"""Declarations become data objects: encoders from tasks and preprocessing, a module from the data section."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from torch.utils.data import Dataset

from src.config import ComponentConfig
from src.core import DatasetInfo, Geometry, Sample
from src.data import DataModule, StandardPreprocessor
from src.data.build import build_data_module, build_preprocessor, build_target_encoder
from src.data.encoders import IdentityEncoder, LabelEncoder, ScalarEncoder
from src.data.table import TableDataModule
from tests.support.declarations import CLASSES, component, preprocessing_config, task_config

PREPROCESSING = preprocessing_config()
task = task_config


class TestTargetEncoder:
    def test_the_kind_default_serves_when_nothing_is_declared(self) -> None:
        encoder = build_target_encoder("species", task(classes=CLASSES), ComponentConfig(name="label"))

        assert isinstance(encoder, LabelEncoder) and encoder.info.num_classes == 2

    def test_a_kind_whose_vocabulary_the_training_split_settles_needs_no_declared_classes(self) -> None:
        """Identities are learned rather than written, so a run naming fifty thousand of them names none.

        This is also what keeps the closed arrangement expressible: a run that wants the vocabulary
        pinned declares `target_encoder: label` with `classes`, and the row below still refuses the pair
        this one leaves out.
        """
        encoder = build_target_encoder("identity", task(), ComponentConfig(name="identity"))

        assert isinstance(encoder, IdentityEncoder)

    def test_a_declared_encoder_wins_over_the_default(self) -> None:
        assert isinstance(
            build_target_encoder("age", task(target_encoder="scalar"), ComponentConfig(name="label")), ScalarEncoder
        )

    def test_classes_may_come_from_a_file(self, tmp_path: Path) -> None:
        (tmp_path / "classes.txt").write_text("cat\n\ndog\n")

        encoder = build_target_encoder(
            "species", task(classes={"file": str(tmp_path / "classes.txt")}), ComponentConfig(name="label")
        )

        assert encoder.info.classes == {0: "cat", 1: "dog"}

    @pytest.mark.parametrize(
        ("declared", "default", "reason"),
        [
            pytest.param(
                {"target_encoder": {"name": "label", "classes": {0: "x"}}},
                "label",
                "on the task",
                id="classes inside the encoder",
            ),
            pytest.param({}, "label", "declare classes", id="vocabulary without classes"),
            pytest.param(
                {"classes": {0: "x"}, "target_encoder": "scalar"}, None, "no vocabulary", id="classes nobody reads"
            ),
            pytest.param({}, None, "no default", id="no encoder at all"),
        ],
    )
    def test_refuses_a_vocabulary_declared_in_the_wrong_place(
        self, declared: dict[str, Any], default: str | None, reason: str
    ) -> None:
        with pytest.raises(ValueError, match=reason):
            build_target_encoder("t", task(**declared), ComponentConfig(name=default) if default else None)


class TestPreprocessor:
    def test_inputs_come_from_preprocessing_and_targets_from_tasks(self) -> None:
        tasks = {"species": task(classes=CLASSES), "age": task(target_column="age", target_encoder="scalar")}

        built = build_preprocessor(PREPROCESSING, tasks, {"species": ComponentConfig(name="label")})

        assert isinstance(built, StandardPreprocessor)
        assert set(built.inputs) == {"image"} and set(built.targets) == {"species", "age"}
        assert built.geometries == {
            "inputs": {"image": Geometry.IMAGE},
            # Published rather than filtered: what a pipeline can carry is the pipeline's to decide.
            "targets": {"species": Geometry.NONE, "age": Geometry.NONE},
            "auxiliary_inputs": {},
        }

    def test_a_task_without_a_target_gets_no_encoder(self) -> None:
        built = build_preprocessor(PREPROCESSING, {"align": task_config(kind="contrastive", target_column=None)}, {})

        assert isinstance(built, StandardPreprocessor) and built.targets == {}

    def test_a_declared_cache_is_attached(self) -> None:
        built = build_preprocessor(preprocessing_config(cache={"name": "ram", "max_gib": 0.01}), {}, {})

        assert isinstance(built, StandardPreprocessor) and built.cache is not None


class TestDataModule:
    @pytest.fixture
    def rows(self, tmp_path: Path) -> Path:
        pd.DataFrame({"image_path": ["a.png"] * 4, "species": ["cat", "dog"] * 2}).to_csv(
            tmp_path / "rows.csv", index=False
        )
        return tmp_path / "rows.csv"

    @pytest.fixture
    def preprocessor(self) -> StandardPreprocessor:
        built = build_preprocessor(PREPROCESSING, {"species": task(classes=CLASSES)}, {"species": component("label")})
        assert isinstance(built, StandardPreprocessor)
        return built

    def test_the_table_grammar_is_translated(self, rows: Path, preprocessor: StandardPreprocessor) -> None:
        declared = component(
            {
                "name": "table",
                "source": str(rows),
                "inputs": {"image": {"column": "image_path"}},
                "split": {"train": 0.5, "val": 0.5, "seed": 1},
            }
        )

        module = build_data_module(declared, preprocessor=preprocessor, targets={"species": "species"}, transforms={})
        module.setup(("train", "val"))

        assert isinstance(module, TableDataModule) and module.info.splits == ("train", "val")

    def test_pre_divided_sources_by_split_name(self, rows: Path, preprocessor: StandardPreprocessor) -> None:
        declared = component(
            {
                "name": "table",
                "source": {"train": str(rows), "val": {"path": str(rows), "format": "csv"}},
                "inputs": {"image": "image_path"},
            }
        )

        module = build_data_module(declared, preprocessor=preprocessor, targets={}, transforms={})
        module.setup(("val",))

        assert module.info.splits == ("val",)

    def test_a_custom_module_receives_the_same_facts(self, preprocessor: StandardPreprocessor) -> None:
        declared = component({"_target_": "tests.unit.data.test_build.Recording", "flag": True})

        module = build_data_module(
            declared, preprocessor=preprocessor, targets={"t": "c"}, transforms={"train": lambda s: s}
        )

        assert (
            isinstance(module, Recording)
            and module.received["flag"] is True
            and module.received["targets"] == {"t": "c"}
        )


class Recording(DataModule):
    """A module of one's own, reached by `_target_`: it records what the builder handed it."""

    def __init__(self, **received: Any) -> None:
        self.received = received

    @property
    def preprocessor(self) -> StandardPreprocessor:
        return cast(StandardPreprocessor, self.received["preprocessor"])

    @property
    def info(self) -> DatasetInfo:
        return self.preprocessor.info

    def setup(self, splits: Sequence[str]) -> None:
        return None

    def dataset(self, split: str) -> Dataset[Sample]:
        raise LookupError(split)
