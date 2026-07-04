"""Dataframe splitting: random/stratified split ratios and max_samples capping."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.enums import Stage
from src.core.runtime import RuntimeContext
from src.data import (
    CsvDataSource,
    DataModule,
    LabelEncoder,
    TargetBinding,
    split_dataframe,
)
from tests.support.builders import make_transform as _make_transform

_LABEL_MAPPING: dict[int, str] = {0: "cat", 1: "cow", 2: "dog"}

LABELS = ["cat", "dog", "cow"]


@pytest.fixture
def csv_path(make_image_csv: Callable[..., Path]) -> Path:
    """15 synthetic 32x32 RGB jpgs across 3 classes and a CSV indexing them."""
    return make_image_csv(count=15, size=32, seed=0, labels=LABELS)


def _binding() -> TargetBinding:
    return TargetBinding(name="label", column="label", encoder=LabelEncoder(class_mapping=_LABEL_MAPPING))


class TestSplitDataframe:
    def _frame(self, n: int = 60) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        labels = (["cat"] * 20 + ["dog"] * 20 + ["cow"] * 20)[:n]
        return pd.DataFrame({"label": labels, "score": rng.random(n)})

    def test_random_split_sizes(self) -> None:
        frame = self._frame()
        parts = split_dataframe(frame, {Stage.TRAIN: 0.7, Stage.VAL: 0.15, Stage.TEST: 0.15}, seed=0)
        assert sum(len(p) for p in parts.values()) == 60

    def test_categorical_stratify_preserves_distribution(self) -> None:
        frame = self._frame()
        parts = split_dataframe(
            frame,
            {Stage.TRAIN: 0.7, Stage.VAL: 0.15, Stage.TEST: 0.15},
            seed=0,
            stratify_column="label",
        )
        for part in parts.values():
            counts = part["label"].value_counts(normalize=True)
            for cls in ["cat", "dog", "cow"]:
                assert abs(counts.get(cls, 0) - 1 / 3) < 0.15

    def test_numeric_stratify_works(self) -> None:
        frame = self._frame()
        parts = split_dataframe(
            frame,
            {Stage.TRAIN: 0.7, Stage.VAL: 0.3},
            seed=0,
            stratify_column="score",
        )
        assert sum(len(p) for p in parts.values()) == 60

    def test_multilabel_stratify_works(self) -> None:
        rng = np.random.default_rng(1)
        labels = [
            ",".join(rng.choice(["a", "b", "c"], size=rng.integers(1, 3), replace=False).tolist()) for _ in range(60)
        ]
        frame = pd.DataFrame({"tags": labels})
        parts = split_dataframe(
            frame,
            {Stage.TRAIN: 0.7, Stage.VAL: 0.3},
            seed=0,
            stratify_column="tags",
        )
        assert sum(len(p) for p in parts.values()) == 60

    def test_missing_stratify_column_raises(self) -> None:
        frame = self._frame()
        with pytest.raises(ValueError, match="stratify_column"):
            split_dataframe(frame, {Stage.TRAIN: 0.8, Stage.VAL: 0.2}, seed=0, stratify_column="nonexistent")


class TestSplit:
    def test_split_sizes_sum_and_are_disjoint(self) -> None:
        frame = pd.DataFrame({"x": range(15)})
        splits = split_dataframe(frame, {Stage.TRAIN: 0.6, Stage.VAL: 0.2, Stage.TEST: 0.2}, seed=42)
        assert sum(len(part) for part in splits.values()) == 15
        assert len(splits[Stage.TRAIN]) == 9
        all_values = pd.concat(splits.values())["x"].tolist()
        assert sorted(all_values) == list(range(15))  # no row lost or duplicated


class TestMaxSamples:
    def test_int_caps_rows(self, csv_path: Path) -> None:
        dm = DataModule(
            target_bindings=[_binding()],
            inputs_config="image_path",
            transforms={s: _make_transform() for s in Stage},
            runtime=RuntimeContext(),
            batch_size=4,
            seed=0,
            source=CsvDataSource(str(csv_path)),
            split={Stage.TRAIN: 0.8, Stage.VAL: 0.2},
            max_samples=6,
        )
        dm.setup()
        total = sum(len(ds) for datasets in dm._datasets.values() for ds in datasets)
        assert total == 6

    def test_float_caps_fraction(self, csv_path: Path) -> None:
        dm = DataModule(
            target_bindings=[_binding()],
            inputs_config="image_path",
            transforms={s: _make_transform() for s in Stage},
            runtime=RuntimeContext(),
            batch_size=4,
            seed=0,
            source=CsvDataSource(str(csv_path)),
            split={Stage.TRAIN: 0.8, Stage.VAL: 0.2},
            max_samples=0.5,
        )
        dm.setup()
        total = sum(len(ds) for datasets in dm._datasets.values() for ds in datasets)
        assert total == 8  # 50% of 15 rows = 7.5 → 8 (pandas rounds up)

    def test_none_keeps_all_rows(self, csv_path: Path) -> None:
        dm = DataModule(
            target_bindings=[_binding()],
            inputs_config="image_path",
            transforms={s: _make_transform() for s in Stage},
            runtime=RuntimeContext(),
            batch_size=4,
            seed=0,
            source=CsvDataSource(str(csv_path)),
            split={Stage.TRAIN: 0.8, Stage.VAL: 0.2},
        )
        dm.setup()
        assert sum(len(ds) for datasets in dm._datasets.values() for ds in datasets) == 15
