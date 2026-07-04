"""Dataset/collate behaviour and DataModule assembly (dataloader kwargs included)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch

from src.core.entities import Sample
from src.core.enums import Stage
from src.core.runtime import RuntimeContext
from src.data import (
    CsvDataSource,
    DataLoaderOptions,
    DataModule,
    LabelEncoder,
    SourceBinding,
    TargetBinding,
    collate_samples,
)
from src.data.datamodule import _build_input_bindings
from src.data.dataset import Dataset
from tests.support.builders import make_transform as _make_transform

_LABEL_MAPPING: dict[int, str] = {0: "cat", 1: "cow", 2: "dog"}

LABELS = ["cat", "dog", "cow"]


@pytest.fixture
def csv_path(make_image_csv: Callable[..., Path]) -> Path:
    """15 synthetic 32x32 RGB jpgs across 3 classes and a CSV indexing them."""
    return make_image_csv(count=15, size=32, seed=0, labels=LABELS)


def _binding() -> TargetBinding:
    return TargetBinding(name="label", column="label", encoder=LabelEncoder(class_mapping=_LABEL_MAPPING))


def _binding_fitted(frame: pd.DataFrame) -> TargetBinding:
    binding = _binding()
    binding.encoder.fit(frame["label"])
    return binding


class TestDatasetAndCollate:
    def test_item_is_image_tensor_and_long_target(self, csv_path: Path) -> None:
        frame = pd.read_csv(csv_path)
        dataset = Dataset(
            frame=frame,
            input_bindings=_build_input_bindings("image_path", frame),
            target_bindings=[_binding_fitted(frame)],
            transform=_make_transform((16, 16)),
        )
        sample = dataset[0]
        assert sample.inputs["image"].shape == (3, 16, 16)
        assert sample.inputs["image"].dtype.is_floating_point
        assert sample.targets["label"].dtype.is_floating_point is False

    def test_collate_stacks_batch(self, csv_path: Path) -> None:
        frame = pd.read_csv(csv_path)
        dataset = Dataset(
            frame=frame,
            input_bindings=_build_input_bindings("image_path", frame),
            target_bindings=[_binding_fitted(frame)],
            transform=_make_transform((16, 16)),
        )
        batch = collate_samples([dataset[i] for i in range(4)])
        assert batch.inputs["image"].shape == (4, 3, 16, 16)
        assert batch.targets["label"].shape == (4,)

    def test_collate_transposes_meta_sources(self) -> None:
        samples = [
            Sample(
                inputs={"image": torch.zeros(3, 4, 4)},
                targets={"mask": torch.zeros(4, 4, dtype=torch.long)},
                meta={
                    "index": i,
                    "input_sources": {"image": f"/img/{i}.jpg"},
                    "target_sources": {"mask": f"/msk/{i}.png"},
                },
            )
            for i in range(3)
        ]
        batch = collate_samples(samples)
        assert batch.meta["index"] == [0, 1, 2]  # scalar → per-sample list
        assert batch.meta["input_sources"] == {
            "image": ["/img/0.jpg", "/img/1.jpg", "/img/2.jpg"]
        }  # dict → dict-of-lists
        assert batch.meta["target_sources"] == {"mask": ["/msk/0.png", "/msk/1.png", "/msk/2.png"]}


class TestDataModule:
    def test_setup_infers_num_classes_and_yields_batch(self, csv_path: Path) -> None:
        runtime = RuntimeContext()
        transforms = {stage: _make_transform((16, 16)) for stage in Stage}
        datamodule = DataModule(
            target_bindings=[_binding()],
            inputs_config="image_path",
            transforms=transforms,
            runtime=runtime,
            batch_size=4,
            seed=0,
            source=CsvDataSource(str(csv_path)),
            split={Stage.TRAIN: 0.6, Stage.VAL: 0.2, Stage.TEST: 0.2},
        )
        datamodule.setup()

        assert runtime.num_classes == {"label": 3}
        assert sum(len(ds) for datasets in datamodule._datasets.values() for ds in datasets) == 15

        batch = next(iter(datamodule.train_dataloader()))
        assert batch.inputs["image"].shape[1:] == (3, 16, 16)
        assert batch.targets["label"].dtype.is_floating_point is False

    def test_presplit_mode_fits_on_train_only(self, csv_path: Path, tmp_path: Path) -> None:
        """Pre-split: separate train/val CSVs; codec fitted on train, applied to val."""

        full_frame = pd.read_csv(csv_path)
        train_csv = tmp_path / "train.csv"
        val_csv = tmp_path / "val.csv"
        full_frame.iloc[:10].to_csv(train_csv, index=False)
        full_frame.iloc[10:].to_csv(val_csv, index=False)

        runtime = RuntimeContext()
        transforms = {s: _make_transform((16, 16)) for s in Stage}
        datamodule = DataModule(
            target_bindings=[_binding()],
            inputs_config="image_path",
            transforms=transforms,
            runtime=runtime,
            batch_size=4,
            seed=0,
            staged_sources={
                Stage.TRAIN: [SourceBinding(CsvDataSource(str(train_csv)), {Stage.TRAIN: transforms[Stage.TRAIN]})],
                Stage.VAL: [SourceBinding(CsvDataSource(str(val_csv)), {Stage.VAL: transforms[Stage.VAL]})],
            },
        )
        datamodule.setup()

        assert runtime.num_classes == {"label": 3}
        assert len(datamodule._datasets[Stage.TRAIN][0]) == 10
        assert len(datamodule._datasets[Stage.VAL][0]) == 5

        batch = next(iter(datamodule.train_dataloader()))
        assert batch.inputs["image"].shape[1:] == (3, 16, 16)

    def test_statistics_returns_per_task_distributions(self, csv_path: Path) -> None:
        """statistics() skips encoders that lack SupportsSummary (behavior matches the old None-skipping)."""
        from src.data.statistics import CategoricalDistribution, SupportsSummary

        runtime = RuntimeContext()
        transforms = {stage: _make_transform((16, 16)) for stage in Stage}
        datamodule = DataModule(
            target_bindings=[_binding()],
            inputs_config="image_path",
            transforms=transforms,
            runtime=runtime,
            batch_size=4,
            seed=0,
            source=CsvDataSource(str(csv_path)),
            split={Stage.TRAIN: 0.6, Stage.VAL: 0.2, Stage.TEST: 0.2},
        )
        datamodule.setup()
        stats = datamodule.statistics()

        # The label encoder supports summarization → the task appears in the output.
        assert "label" in stats
        assert isinstance(datamodule._target_bindings[0].encoder, SupportsSummary)
        for stage, distribution in stats["label"].items():
            assert isinstance(distribution, CategoricalDistribution)
            # All three classes present across every stage.
            assert set(distribution.counts.keys()) == {"cat", "cow", "dog"}

    def test_statistics_skips_non_summarizable_encoders(self, csv_path: Path) -> None:
        """An encoder without SupportsSummary (e.g. a stub) is silently omitted."""
        from src.data.encoders import TargetEncoder
        from src.data.statistics import SupportsSummary

        class _PassthroughEncoder(TargetEncoder):
            def fit(self, values: Iterable[Any]) -> None:
                pass

            def load(self, value: Any) -> Any:
                return value

            def to_tensor(self, value: Any) -> torch.Tensor:
                return torch.tensor(0)

        runtime = RuntimeContext()
        transforms = {stage: _make_transform((16, 16)) for stage in Stage}
        passthrough_binding = TargetBinding(name="extra", column="label", encoder=_PassthroughEncoder())
        datamodule = DataModule(
            target_bindings=[_binding(), passthrough_binding],
            inputs_config="image_path",
            transforms=transforms,
            runtime=runtime,
            batch_size=4,
            seed=0,
            source=CsvDataSource(str(csv_path)),
            split={Stage.TRAIN: 0.6, Stage.VAL: 0.2, Stage.TEST: 0.2},
        )
        datamodule.setup()
        stats = datamodule.statistics()

        assert "label" in stats  # summarizable encoder is present
        assert "extra" not in stats  # non-summarizable encoder is absent
        assert not isinstance(passthrough_binding.encoder, SupportsSummary)


class TestDataLoaderKwargs:
    def _dm(self, csv_path: Path, **extra: object) -> DataModule:
        return DataModule(
            target_bindings=[_binding()],
            inputs_config="image_path",
            transforms={s: _make_transform() for s in Stage},
            runtime=RuntimeContext(),
            batch_size=4,
            seed=0,
            source=CsvDataSource(str(csv_path)),
            split={Stage.TRAIN: 0.8, Stage.VAL: 0.2},
            dataloader_options=DataLoaderOptions(extra_kwargs=dict(extra)),
        )

    def test_extra_kwargs_reach_dataloader(self, csv_path: Path) -> None:
        dm = self._dm(csv_path, timeout=5)
        dm.setup()
        assert dm.train_dataloader().timeout == 5

    def test_framework_keys_win_over_extras(self, csv_path: Path) -> None:
        # Defensive merge: even a framework-owned key in extras must not override the real value.
        dm = self._dm(csv_path, batch_size=999)
        dm.setup()
        assert dm.train_dataloader().batch_size == 4
