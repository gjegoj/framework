"""Tests for the src/reporting package: DistributionRenderer registry + box-plot dispatch."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from src.core.enums import Stage
from src.core.plotting import BoxPlot, Plot
from src.core.ports import HistogramLogger, PlotLogger
from src.data.statistics import (
    CategoricalDistribution,
    ContinuousDistribution,
    Histogram,
)
from src.reporting import report_dataset_statistics
from src.reporting.renderers import (
    CategoricalDistributionRenderer,
    ContinuousDistributionRenderer,
)

# ------------------------------------------------------------------ test double


class CapturingPlotLogger(HistogramLogger, PlotLogger):
    """Records every ``log_histogram`` and ``log_plot`` call for assertions.

    Implements only the two artifact ports the distribution renderers use — Interface
    Segregation means this double no longer stubs matrix / curve / html / single-value.
    """

    def __init__(self) -> None:
        self.histogram_calls: list[dict[str, Any]] = []
        self.plot_calls: list[Plot] = []

    def log_histogram(
        self,
        title: str,
        series: str,
        values: Sequence[float],
        labels: list[str] | None = None,
    ) -> None:
        self.histogram_calls.append({"title": title, "series": series, "values": list(values), "labels": labels})

    def log_plot(self, plot: Plot) -> None:
        self.plot_calls.append(plot)


# ----------------------------------------------------------------- fixtures


def _categorical_per_stage() -> dict[Stage, CategoricalDistribution]:
    return {
        Stage.TRAIN: CategoricalDistribution({"cat": 6, "dog": 4}),
        Stage.VAL: CategoricalDistribution({"cat": 3, "dog": 2}),
    }


def _continuous_per_stage() -> dict[Stage, ContinuousDistribution]:
    return {
        Stage.TRAIN: ContinuousDistribution(
            count=10,
            mean=2.0,
            std=1.0,
            minimum=0.0,
            q25=1.0,
            median=2.0,
            q75=3.0,
            maximum=4.0,
            histogram=Histogram(counts=(4, 6), edges=(0.0, 2.0, 4.0)),
        ),
        Stage.VAL: ContinuousDistribution(
            count=5,
            mean=3.0,
            std=0.5,
            minimum=1.0,
            q25=2.5,
            median=3.0,
            q75=3.5,
            maximum=5.0,
            histogram=Histogram(counts=(2, 3), edges=(1.0, 3.0, 5.0)),
        ),
    }


# ----------------------------------------------------------------- renderer tests


class TestCategoricalDistributionRenderer:
    def test_log_emits_grouped_histogram_per_stage(self) -> None:
        logger = CapturingPlotLogger()
        renderer = CategoricalDistributionRenderer()
        renderer.log("animals", _categorical_per_stage(), logger)  # type: ignore[arg-type]

        titles = {call["title"] for call in logger.histogram_calls}
        assert titles == {"dataset/animals"}

        series_values = {call["series"] for call in logger.histogram_calls}
        assert series_values == {"train", "val"}

    def test_log_histogram_values_and_labels(self) -> None:
        logger = CapturingPlotLogger()
        renderer = CategoricalDistributionRenderer()
        renderer.log("animals", _categorical_per_stage(), logger)  # type: ignore[arg-type]

        train_call = next(c for c in logger.histogram_calls if c["series"] == "train")
        assert train_call["values"] == [6.0, 4.0]
        assert train_call["labels"] == ["cat", "dog"]

    def test_log_emits_no_plot_calls(self) -> None:
        logger = CapturingPlotLogger()
        CategoricalDistributionRenderer().log("animals", _categorical_per_stage(), logger)  # type: ignore[arg-type]
        assert logger.plot_calls == []

    def test_table_contains_class_names(self) -> None:
        renderer = CategoricalDistributionRenderer()
        per_stage = _categorical_per_stage()
        table = renderer.table("animals", per_stage)  # type: ignore[arg-type]
        # Rich Table stores column headers; check at least the title is set.
        assert "animals" in (table.title or "")

    def test_table_is_identical_to_standalone_builder(self) -> None:
        from src.reporting.tables import categorical_table

        per_stage = _categorical_per_stage()
        stages = [Stage.TRAIN, Stage.VAL]
        expected = categorical_table("animals", per_stage, stages)
        renderer = CategoricalDistributionRenderer()
        result = renderer.table("animals", per_stage)  # type: ignore[arg-type]
        assert result.title == expected.title
        assert result.row_count == expected.row_count


class TestContinuousDistributionRenderer:
    def test_log_emits_exactly_one_plot(self) -> None:
        logger = CapturingPlotLogger()
        renderer = ContinuousDistributionRenderer()
        renderer.log("score", _continuous_per_stage(), logger)  # type: ignore[arg-type]

        assert len(logger.plot_calls) == 1

    def test_log_plot_is_box_plot_type(self) -> None:
        logger = CapturingPlotLogger()
        ContinuousDistributionRenderer().log("score", _continuous_per_stage(), logger)  # type: ignore[arg-type]
        assert isinstance(logger.plot_calls[0], BoxPlot)

    def test_log_box_plot_title(self) -> None:
        logger = CapturingPlotLogger()
        ContinuousDistributionRenderer().log("score", _continuous_per_stage(), logger)  # type: ignore[arg-type]
        assert logger.plot_calls[0].title == "dataset/score"

    def test_log_box_plot_categories_match_stages(self) -> None:
        logger = CapturingPlotLogger()
        ContinuousDistributionRenderer().log("score", _continuous_per_stage(), logger)  # type: ignore[arg-type]
        assert logger.plot_calls[0].categories == ["train", "val"]

    def test_log_box_plot_boxes_mirror_per_stage_quartiles(self) -> None:
        logger = CapturingPlotLogger()
        ContinuousDistributionRenderer().log("score", _continuous_per_stage(), logger)  # type: ignore[arg-type]
        boxes = logger.plot_calls[0].boxes
        assert len(boxes) == 2

        train_box = boxes[0]
        assert train_box.minimum == 0.0
        assert train_box.q25 == 1.0
        assert train_box.median == 2.0
        assert train_box.q75 == 3.0
        assert train_box.maximum == 4.0
        assert train_box.mean == 2.0

        val_box = boxes[1]
        assert val_box.minimum == 1.0
        assert val_box.q25 == 2.5
        assert val_box.median == 3.0
        assert val_box.q75 == 3.5
        assert val_box.maximum == 5.0
        assert val_box.mean == 3.0

    def test_log_emits_no_histogram_calls(self) -> None:
        logger = CapturingPlotLogger()
        ContinuousDistributionRenderer().log("score", _continuous_per_stage(), logger)  # type: ignore[arg-type]
        assert logger.histogram_calls == []

    def test_table_contains_task_name(self) -> None:
        renderer = ContinuousDistributionRenderer()
        per_stage = _continuous_per_stage()
        table = renderer.table("score", per_stage)  # type: ignore[arg-type]
        assert "score" in (table.title or "")


# ----------------------------------------------------------------- dispatcher tests


class TestReportDatasetStatistics:
    def _statistics(self) -> dict[str, dict[Stage, CategoricalDistribution | ContinuousDistribution]]:
        label: dict[Stage, CategoricalDistribution | ContinuousDistribution] = dict(_categorical_per_stage())
        score: dict[Stage, CategoricalDistribution | ContinuousDistribution] = dict(_continuous_per_stage())
        return {"label": label, "score": score}

    def test_routes_categorical_task_to_histogram(self) -> None:
        logger = CapturingPlotLogger()
        report_dataset_statistics(self._statistics(), logger)

        histogram_titles = {call["title"] for call in logger.histogram_calls}
        assert "dataset/label" in histogram_titles

    def test_routes_continuous_task_to_box_plot(self) -> None:
        logger = CapturingPlotLogger()
        report_dataset_statistics(self._statistics(), logger)

        assert any(isinstance(plot, BoxPlot) and plot.title == "dataset/score" for plot in logger.plot_calls)

    def test_prints_table_for_each_task(self, capsys: Any) -> None:
        # Tables render to a Rich Console — just verify the call completes without error.
        report_dataset_statistics(self._statistics(), object())

    def test_no_log_calls_without_plot_logger(self) -> None:
        # Should not raise even with a plain object as logger.
        report_dataset_statistics(self._statistics(), object())

    def test_empty_statistics_is_noop(self) -> None:
        report_dataset_statistics({}, CapturingPlotLogger())

    def test_unregistered_distribution_type_raises(self) -> None:
        class UnknownDistribution:
            pass

        statistics: dict[str, dict[Stage, CategoricalDistribution | ContinuousDistribution]] = {
            "task": {Stage.TRAIN: UnknownDistribution()}  # type: ignore[dict-item]
        }
        with pytest.raises(KeyError, match="distribution_renderer"):
            report_dataset_statistics(statistics, object())
