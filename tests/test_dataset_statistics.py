"""Tests for dataset distribution statistics: summarize → DatasetStatistics → report."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.core.enums import Stage
from src.core.runtime import RuntimeContext
from src.data import (
    CacheOptions,
    CategoricalDistribution,
    ContinuousDistribution,
    CsvDataSource,
    DataModule,
    LabelEncoder,
    MaskEncoder,
    MultiLabelEncoder,
    ScalarEncoder,
    TargetBinding,
)
from src.reporting import report_dataset_statistics
from tests.test_data import _make_transform
from tests.test_metrics import FakePlotLogger


class TestEntities:
    def test_categorical_total_and_relative(self) -> None:
        distribution = CategoricalDistribution(counts={"a": 3, "b": 1, "c": 0})
        assert distribution.total == 4
        assert distribution.relative["a"] == pytest.approx(0.75)
        assert distribution.relative["c"] == 0.0

    def test_categorical_relative_zero_total(self) -> None:
        assert CategoricalDistribution(counts={"a": 0}).relative["a"] == 0.0


class TestEncoderSummarize:
    def test_label_counts_in_class_order_with_zeros(self) -> None:
        encoder = LabelEncoder(class_mapping={0: "cat", 1: "dog", 2: "cow"})
        distribution = encoder.summarize(pd.Series(["cat", "dog", "cat", "cat"]))
        assert isinstance(distribution, CategoricalDistribution)
        assert distribution.counts == {"cat": 3, "dog": 1, "cow": 0}  # index order, zero class kept

    def test_multilabel_counts_occurrences(self) -> None:
        encoder = MultiLabelEncoder(separator="|", class_mapping={0: "a", 1: "b", 2: "c"})
        distribution = encoder.summarize(pd.Series(["a|b", "b", "c", "a|b|c"]))
        assert isinstance(distribution, CategoricalDistribution)
        assert distribution.counts == {"a": 2, "b": 3, "c": 2}

    def test_scalar_summary(self) -> None:
        distribution = ScalarEncoder().summarize(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert isinstance(distribution, ContinuousDistribution)
        assert distribution.count == 5
        assert distribution.mean == pytest.approx(3.0)
        assert distribution.median == pytest.approx(3.0)
        assert (distribution.minimum, distribution.maximum) == (1.0, 5.0)

    def test_scalar_drops_nan(self) -> None:
        distribution = ScalarEncoder().summarize(pd.Series([1.0, float("nan"), 3.0]))
        assert isinstance(distribution, ContinuousDistribution)
        assert distribution.count == 2

    def test_scalar_empty_is_none(self) -> None:
        assert ScalarEncoder().summarize(pd.Series([], dtype=float)) is None

    def test_mask_encoder_is_not_summarizable(self) -> None:
        """Segmentation is deferred: MaskEncoder does not implement SupportsSummary (omitted from the report)."""
        from src.data.statistics import SupportsSummary

        assert not isinstance(MaskEncoder(class_mapping={0: "bg", 1: "fg"}), SupportsSummary)

    def test_caching_wrapper_forwards_summarize(self) -> None:
        """A cached (spatial) encoder must keep its distribution — so segmentation drops in later."""
        from src.data.cache import ArrayCache, caching_target_encoder
        from src.data.statistics import SupportsSummary

        inner = LabelEncoder(class_mapping={0: "cat", 1: "dog"})
        wrapped = caching_target_encoder(inner, ArrayCache(max_bytes=1024))
        assert isinstance(wrapped, SupportsSummary)
        distribution = wrapped.summarize(pd.Series(["cat", "dog", "cat"]))
        assert isinstance(distribution, CategoricalDistribution)
        assert distribution.counts == {"cat": 2, "dog": 1}


def _datamodule(tmp_path: Path) -> DataModule:
    """Split-mode DataModule over a synthetic CSV (label + score + a mask-typed task)."""
    rows = [
        {"image_path": f"{index}.jpg", "label": ["cat", "dog", "cow"][index % 3], "score": float(index)}
        for index in range(12)
    ]
    csv = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    bindings = [
        TargetBinding(name="label", column="label", encoder=LabelEncoder(class_mapping={0: "cat", 1: "dog", 2: "cow"})),
        TargetBinding(name="score", column="score", encoder=ScalarEncoder()),
        TargetBinding(name="seg", column="image_path", encoder=MaskEncoder(class_mapping={0: "bg", 1: "fg"})),
    ]
    return DataModule(
        target_bindings=bindings,
        inputs_config="image_path",
        transforms={stage: _make_transform() for stage in (Stage.TRAIN, Stage.VAL, Stage.TEST)},
        runtime=RuntimeContext(),
        batch_size=2,
        source=CsvDataSource(str(csv)),
        split={Stage.TRAIN: 0.5, Stage.VAL: 0.25, Stage.TEST: 0.25},
        cache_options=CacheOptions(max_bytes=0),  # cache off: no file reads at setup
    )


class TestDataModuleStatistics:
    def test_requires_setup(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="setup"):
            _datamodule(tmp_path).statistics()

    def test_per_task_distributions_and_mask_omitted(self, tmp_path: Path) -> None:
        data_module = _datamodule(tmp_path)
        data_module.setup()
        statistics = data_module.statistics()

        assert set(statistics) == {"label", "score"}  # 'seg' (mask) omitted — summarize deferred
        assert all(isinstance(value, CategoricalDistribution) for value in statistics["label"].values())
        assert all(isinstance(value, ContinuousDistribution) for value in statistics["score"].values())

    def test_counts_sum_to_dataset_size_across_stages(self, tmp_path: Path) -> None:
        data_module = _datamodule(tmp_path)
        data_module.setup()
        statistics = data_module.statistics()
        label = [value for value in statistics["label"].values() if isinstance(value, CategoricalDistribution)]
        score = [value for value in statistics["score"].values() if isinstance(value, ContinuousDistribution)]
        assert sum(distribution.total for distribution in label) == 12
        assert sum(distribution.count for distribution in score) == 12


class TestReporter:
    def _statistics(self) -> dict[str, dict[Stage, object]]:
        return {
            "label": {
                Stage.TRAIN: CategoricalDistribution({"cat": 6, "dog": 4}),
                Stage.TEST: CategoricalDistribution({"cat": 3, "dog": 2}),
            },
            "score": {
                Stage.TRAIN: ContinuousDistribution(
                    count=10,
                    mean=2.0,
                    std=1.0,
                    minimum=0.0,
                    q25=1.0,
                    median=2.0,
                    q75=3.0,
                    maximum=4.0,
                ),
            },
        }

    def test_logs_one_histogram_per_task_and_stage(self) -> None:
        logger = FakePlotLogger()
        report_dataset_statistics(self._statistics(), logger)  # type: ignore[arg-type]
        # Categorical: one grouped plot (series per stage); continuous: one box plot via log_plot.
        assert {call["title"] for call in logger.histogram_calls} == {"dataset/label"}
        label_series = {call["series"] for call in logger.histogram_calls if call["title"] == "dataset/label"}
        assert label_series == {"train", "test"}
        # Continuous task emits a box plot instead of histograms.
        assert len(logger.plot_calls) == 1
        assert logger.plot_calls[0].title == "dataset/score"

    def test_categorical_histogram_values_and_labels(self) -> None:
        logger = FakePlotLogger()
        report_dataset_statistics(self._statistics(), logger)  # type: ignore[arg-type]
        train = next(c for c in logger.histogram_calls if c["title"] == "dataset/label" and c["series"] == "train")
        assert train["values"] == [6.0, 4.0]
        assert train["labels"] == ["cat", "dog"]

    def test_continuous_box_plot_quartiles(self) -> None:
        # Continuous distributions now log a box plot instead of per-stage histograms.
        from src.core.plotting import BoxPlot

        logger = FakePlotLogger()
        report_dataset_statistics(self._statistics(), logger)  # type: ignore[arg-type]
        assert len(logger.plot_calls) == 1
        box_plot = logger.plot_calls[0]
        assert isinstance(box_plot, BoxPlot)
        assert box_plot.categories == ["train"]
        assert box_plot.boxes[0].minimum == 0.0
        assert box_plot.boxes[0].q25 == 1.0
        assert box_plot.boxes[0].median == 2.0
        assert box_plot.boxes[0].q75 == 3.0
        assert box_plot.boxes[0].maximum == 4.0
        assert box_plot.boxes[0].mean == 2.0

    def test_noop_without_plot_logger(self) -> None:
        report_dataset_statistics(self._statistics(), object())  # type: ignore[arg-type]  # prints, no logging

    def test_empty_statistics_is_noop(self) -> None:
        report_dataset_statistics({}, FakePlotLogger())


class TestDatasetStatsCallback:
    """Thin lifecycle glue: report once, before the first stage, guarding rank + datamodule type."""

    def _trainer(self, datamodule: object, *, global_zero: bool = True) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(is_global_zero=global_zero, datamodule=datamodule, logger=None)

    def _lit_datamodule(self, tmp_path: Path) -> object:
        from src.training.modules import LitDataModule

        inner = _datamodule(tmp_path)
        inner.setup()
        return LitDataModule(inner)

    def test_registered(self) -> None:
        from src.callbacks.registry import callback_registry

        assert "dataset_stats" in callback_registry

    def test_reports_once_across_fit_and_test(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.callbacks.dataset_stats import DatasetStatsCallback

        reported: list[Any] = []
        monkeypatch.setattr(
            "src.callbacks.dataset_stats.report_dataset_statistics",
            lambda statistics, logger: reported.append(statistics),
        )
        callback, trainer = DatasetStatsCallback(), self._trainer(self._lit_datamodule(tmp_path))
        callback.on_fit_start(trainer, MagicMock())
        callback.on_test_start(trainer, MagicMock())  # run-once guard → no second report
        assert len(reported) == 1
        assert set(reported[0]) == {"label", "score"}  # the computed stats reached the renderer

    def test_skips_non_global_zero(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.callbacks.dataset_stats import DatasetStatsCallback

        reported: list[object] = []
        monkeypatch.setattr("src.callbacks.dataset_stats.report_dataset_statistics", lambda *_: reported.append(1))
        trainer = self._trainer(self._lit_datamodule(tmp_path), global_zero=False)
        DatasetStatsCallback().on_fit_start(trainer, MagicMock())
        assert reported == []

    def test_skips_when_datamodule_cannot_report(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.callbacks.dataset_stats import DatasetStatsCallback

        reported: list[object] = []
        monkeypatch.setattr("src.callbacks.dataset_stats.report_dataset_statistics", lambda *_: reported.append(1))
        DatasetStatsCallback().on_fit_start(self._trainer(object()), MagicMock())  # plain datamodule
        assert reported == []
