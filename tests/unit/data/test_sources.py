"""Data sources (CSV/JSON) and per-source transform overrides."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.core.enums import Stage
from src.core.runtime import RuntimeContext
from src.data import (
    CsvDataSource,
    DataModule,
    JsonDataSource,
    LabelEncoder,
    SourceBinding,
    TargetBinding,
)
from tests.support.builders import make_transform as _make_transform

_LABEL_MAPPING: dict[int, str] = {0: "cat", 1: "cow", 2: "dog"}

LABELS = ["cat", "dog", "cow"]


def _binding() -> TargetBinding:
    return TargetBinding(name="label", column="label", encoder=LabelEncoder(class_mapping=_LABEL_MAPPING))


class TestDataSources:
    def test_csv_reads_and_concatenates(self, tmp_path: Path) -> None:
        a = tmp_path / "a.csv"
        b = tmp_path / "b.csv"
        pd.DataFrame({"x": [1, 2]}).to_csv(a, index=False)
        pd.DataFrame({"x": [3]}).to_csv(b, index=False)
        frame = CsvDataSource([str(a), str(b)]).read()
        assert frame["x"].tolist() == [1, 2, 3]

    def test_json_reads_array_of_records(self, tmp_path: Path) -> None:
        path = tmp_path / "data.json"
        path.write_text('[{"image_path": "a.jpg", "label": "cat"}, {"image_path": "b.jpg", "label": "dog"}]')
        frame = JsonDataSource(str(path)).read()
        assert frame["label"].tolist() == ["cat", "dog"]

    def test_json_concatenates_multiple_files(self, tmp_path: Path) -> None:
        p1 = tmp_path / "1.json"
        p2 = tmp_path / "2.json"
        p1.write_text('[{"x": 1}]')
        p2.write_text('[{"x": 2}, {"x": 3}]')
        frame = JsonDataSource([str(p1), str(p2)]).read()
        assert frame["x"].tolist() == [1, 2, 3]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Data source file not found"):
            CsvDataSource(str(tmp_path / "nope.csv")).read()

    def test_empty_paths_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one path"):
            JsonDataSource([])


class TestPerSourceTransforms:
    @staticmethod
    def _make_csv(tmp_path: Path, name: str, count: int) -> Path:
        image_dir = tmp_path / name
        image_dir.mkdir()
        rng = np.random.default_rng(0)
        rows = []
        for index in range(count):
            array = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
            path = image_dir / f"{index}.jpg"
            cv2.imwrite(str(path), array)
            rows.append({"image_path": str(path), "label": LABELS[index % 3]})
        csv = tmp_path / f"{name}.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        return csv

    def test_override_applies_per_source_and_stage(self, tmp_path: Path) -> None:
        """Source 2's train override (8x8) hits only its train rows; source 1 and every
        val/test row keep the global 16x16 transform."""
        csv1 = self._make_csv(tmp_path, "a", 6)
        csv2 = self._make_csv(tmp_path, "b", 6)
        global_transforms = {stage: _make_transform((16, 16)) for stage in Stage}
        override = {**global_transforms, Stage.TRAIN: _make_transform((8, 8))}  # train differs; val/test global
        dm = DataModule(
            target_bindings=[_binding()],
            inputs_config="image_path",
            transforms=global_transforms,
            runtime=RuntimeContext(),
            batch_size=2,
            seed=0,
            source_bindings=[
                SourceBinding(source=CsvDataSource(str(csv1)), transforms=global_transforms),
                SourceBinding(source=CsvDataSource(str(csv2)), transforms=override),
            ],
            split={Stage.TRAIN: 0.5, Stage.VAL: 0.5},
        )
        dm.setup()

        train = dm._datasets[Stage.TRAIN]
        assert len(train) == 2  # one Dataset per source, combined via ConcatDataset
        assert train[0][0].inputs["image"].shape == (3, 16, 16)  # source 1 train → global
        assert train[1][0].inputs["image"].shape == (3, 8, 8)  # source 2 train → override
        assert dm._datasets[Stage.VAL][1][0].inputs["image"].shape == (3, 16, 16)  # source 2 val → global fallback

    def test_presplit_override_applies_per_source(self, tmp_path: Path) -> None:
        """Pre-split: one train source with its own transform (8x8) applies only to that source;
        the other train source and val keep the global 16x16."""
        csv_a = self._make_csv(tmp_path, "train_a", 4)
        csv_b = self._make_csv(tmp_path, "train_b", 4)
        csv_val = self._make_csv(tmp_path, "val", 4)
        t16, t8 = _make_transform((16, 16)), _make_transform((8, 8))
        dm = DataModule(
            target_bindings=[_binding()],
            inputs_config="image_path",
            transforms={stage: t16 for stage in Stage},
            runtime=RuntimeContext(),
            batch_size=2,
            seed=0,
            staged_sources={
                Stage.TRAIN: [
                    SourceBinding(CsvDataSource(str(csv_a)), {Stage.TRAIN: t16}),
                    SourceBinding(CsvDataSource(str(csv_b)), {Stage.TRAIN: t8}),  # this source → 8x8
                ],
                Stage.VAL: [SourceBinding(CsvDataSource(str(csv_val)), {Stage.VAL: t16})],
            },
        )
        dm.setup()

        train = dm._datasets[Stage.TRAIN]
        assert len(train) == 2  # one Dataset per train source
        assert train[0][0].inputs["image"].shape == (3, 16, 16)  # source a → global
        assert train[1][0].inputs["image"].shape == (3, 8, 8)  # source b → its override
        assert dm._datasets[Stage.VAL][0][0].inputs["image"].shape == (3, 16, 16)  # val → global
