"""MetricReporter epoch flow, metric directions, and the summary metric set."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch

from src.core.entities import Task
from src.core.enums import Stage
from src.metrics.directions import task_metric_directions
from src.metrics.handlers import (
    MetricHandler,
)
from src.metrics.reporter import MetricReporter
from src.tasks.presets import classification, regression
from tests.support.fakes import FakePlotLogger


class TestMetricReporter:
    @staticmethod
    def _task(metric_set: Any) -> Task:
        task = MagicMock()
        task.name = "cls"
        task.class_names = None
        task.metrics = {Stage.VAL: metric_set}
        return cast(Task, task)

    def test_report_logs_each_metric_by_shape_then_resets(self) -> None:
        metric_set = MagicMock()
        metric_set.compute.return_value = {"acc": torch.tensor(0.9), "f1": torch.tensor([0.5, 0.7])}
        log_scalar = MagicMock()

        MetricReporter().report(self._task(metric_set), Stage.VAL, log_scalar, logger=None, step=3)

        keys = [call.args[0] for call in log_scalar.call_args_list]
        assert "cls/acc/val" in keys  # scalar handler
        assert "cls/f1/val/mean" in keys  # vector handler (mean + per-class)
        metric_set.reset.assert_called_once()

    def test_report_routes_matrix_to_plot_logger(self) -> None:
        metric_set = MagicMock()
        metric_set.compute.return_value = {"cm": torch.zeros(3, 3)}
        fake = FakePlotLogger()

        MetricReporter().report(self._task(metric_set), Stage.VAL, MagicMock(), logger=fake, step=1)

        assert len(fake.matrix_calls) == 1

    def test_report_uses_injected_handlers(self) -> None:
        metric_set = MagicMock()
        metric_set.compute.return_value = {"x": torch.tensor(1.0)}
        custom = MagicMock(spec=MetricHandler)
        custom.can_handle.return_value = True

        MetricReporter(handlers=(custom,)).report(self._task(metric_set), Stage.VAL, MagicMock(), None, 0)

        custom.handle.assert_called_once()


class TestTaskMetricDirections:
    def test_keys_use_task_metric_stage_convention(self) -> None:
        task = classification("label", num_classes=3)
        directions = task_metric_directions([task])
        assert "label/f1/train" in directions
        assert "label/f1/val" in directions

    def test_higher_is_better_metrics_map_to_true(self) -> None:
        task = classification("label", num_classes=3)
        directions = task_metric_directions([task])
        assert directions["label/f1/train"] is True
        assert directions["label/precision/val"] is True

    def test_directionless_metrics_map_to_none(self) -> None:
        task = classification("label", num_classes=3)
        directions = task_metric_directions([task])
        assert directions["label/confusion_matrix/train"] is None

    def test_lower_is_better_metric_maps_to_false(self) -> None:
        task = regression("depth", num_classes=1)
        directions = task_metric_directions([task])
        assert directions["depth/mae/train"] is False

    def test_covers_every_stage(self) -> None:
        task = classification("label", num_classes=3)
        directions = task_metric_directions([task])
        stages = {key.rsplit("/", 1)[-1] for key in directions}
        assert stages == {str(Stage.TRAIN), str(Stage.VAL), str(Stage.TEST)}

    def test_multiple_tasks_namespaced_separately(self) -> None:
        directions = task_metric_directions([classification("a", num_classes=2), classification("b", num_classes=2)])
        assert "a/f1/train" in directions
        assert "b/f1/train" in directions


class TestSummaryMetrics:
    """Headline selection for the end-of-run single-value summary."""

    def _metrics(self) -> dict[str, torch.Tensor]:
        return {
            "species/f1/test": torch.tensor(0.75),  # scalar — keep
            "breed/f1/test/mean": torch.tensor(0.16),  # vector mean — keep
            "breed/f1/test/Abyssinian": torch.tensor(0.22),  # per-class — drop
            "mask/iou/test/mean": torch.tensor(0.53),  # vector mean — keep
            "loss/test/total": torch.tensor(4.87),  # stage total loss — keep
            "loss/test/breed/cross_entropy": torch.tensor(3.29),  # loss component — drop
            "species/f1/val": torch.tensor(0.78),  # other stage — drop
        }

    def test_names_match_the_training_table_rows(self) -> None:
        """Stage and the ``mean`` leaf are stripped → ``species/f1``, ``loss/total``, …."""
        from src.metrics.summary import summary_metrics

        selected = summary_metrics(self._metrics(), Stage.TEST)
        assert set(selected) == {"species/f1", "breed/f1", "mask/iou", "loss/total"}

    def test_vector_uses_mean_not_per_class(self) -> None:
        from src.metrics.summary import summary_metrics

        selected = summary_metrics(self._metrics(), Stage.TEST)
        assert selected["breed/f1"] == pytest.approx(0.16, abs=1e-4)  # the mean, not the 0.22 class

    def test_values_are_plain_floats(self) -> None:
        from src.metrics.summary import summary_metrics

        selected = summary_metrics(self._metrics(), Stage.TEST)
        assert all(isinstance(value, float) for value in selected.values())
        assert selected["loss/total"] == pytest.approx(4.87, abs=1e-4)

    def test_other_stage_excluded(self) -> None:
        """``species/f1`` holds the test value (0.75), never the val value (0.78)."""
        from src.metrics.summary import summary_metrics

        assert summary_metrics(self._metrics(), Stage.TEST)["species/f1"] == pytest.approx(0.75, abs=1e-4)

    def test_empty_when_stage_absent(self) -> None:
        from src.metrics.summary import summary_metrics

        assert summary_metrics(self._metrics(), Stage.TRAIN) == {}
