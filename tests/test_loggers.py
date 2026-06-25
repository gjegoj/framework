"""Unit tests for the ClearML logger adapter.

- ``_split_metric_name``: pure static method; runs without clearml installed.
- ``log_plot``: verified via a stub logger; runs without clearml installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import plotly.graph_objects as go
import pytest

from src.core.plotting import BoxPlot, BoxStats
from src.loggers.clearml import ClearMLLogger

split = ClearMLLogger._split_metric_name


class TestSplitMetricName:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            # Losses: stage in the middle → pulled out as the series so train/val/test
            # of the same loss share one title (graph).
            ("loss/train/total", ("loss/total", "train")),
            ("loss/val/total", ("loss/total", "val")),
            ("loss/test/total", ("loss/total", "test")),
            # Composite-loss components keep a per-component graph, grouped by stage.
            ("loss/train/mask/cross_entropy", ("loss/mask/cross_entropy", "train")),
            ("loss/val/mask/cross_entropy", ("loss/mask/cross_entropy", "val")),
            # Metrics are UNCHANGED (last segment = series) — losses are the only exception.
            # Averaged metric: stage trails → train/val/test group on one graph.
            ("species/f1/val", ("species/f1", "val")),
            ("species/f1/train", ("species/f1", "train")),
            # Per-class metric: class name trails → all classes on one per-stage graph
            # (this is the case that must NOT be regrouped by stage).
            ("breed/f1/train/Abyssinian", ("breed/f1/train", "Abyssinian")),
            ("breed/f1/train/Bengal", ("breed/f1/train", "Bengal")),
            # A task literally containing a stage word in a non-loss key stays default.
            ("foo/bar/baz", ("foo/bar", "baz")),
            # Single segment → "value" series.
            ("lr", ("lr", "value")),
        ],
    )
    def test_split(self, name: str, expected: tuple[str, str]) -> None:
        assert split(name) == expected

    def test_metrics_are_not_regrouped_by_stage(self) -> None:
        """Per-class metrics for a stage stay on ONE graph (the regression we fixed)."""
        titles = {split(f"breed/f1/train/{cls}")[0] for cls in ("Abyssinian", "Bengal", "Birman")}
        assert titles == {"breed/f1/train"}

    def test_train_and_val_share_one_title(self) -> None:
        """The whole point: a loss's train/val/test land on the same graph (title)."""
        titles = {split(f"loss/{stage}/total")[0] for stage in ("train", "val", "test")}
        assert titles == {"loss/total"}


class TestClearMLLoggerLogPlot:
    """Verify ClearMLLogger.log_plot delegates to report_plotly exactly once."""

    def _make_logger_with_stub(self) -> tuple[ClearMLLogger, MagicMock]:
        """Construct a ``ClearMLLogger`` with a stub ClearML logger (no real Task)."""
        stub_backend_logger = MagicMock()
        stub_task = MagicMock()
        stub_task.get_logger.return_value = stub_backend_logger

        with patch("clearml.Task.init", return_value=stub_task):
            logger = ClearMLLogger.__new__(ClearMLLogger)
            logger._task = stub_task
            logger._clearml_logger = stub_backend_logger

        return logger, stub_backend_logger

    def test_log_plot_calls_report_plotly_once(self) -> None:
        logger, stub_backend_logger = self._make_logger_with_stub()

        plot = BoxPlot(
            title="score_regression",
            categories=["train", "val"],
            boxes=[
                BoxStats(minimum=0.0, q25=1.0, median=2.0, q75=3.0, maximum=4.0, mean=2.0),
                BoxStats(minimum=0.5, q25=1.5, median=2.5, q75=3.5, maximum=4.5, mean=2.5),
            ],
        )
        logger.log_plot(plot)

        stub_backend_logger.report_plotly.assert_called_once()
        call_kwargs = stub_backend_logger.report_plotly.call_args
        assert call_kwargs.kwargs["title"] == "score_regression"
        assert call_kwargs.kwargs["series"] == ""
        assert isinstance(call_kwargs.kwargs["figure"], go.Figure)
        assert call_kwargs.kwargs["iteration"] == 0
