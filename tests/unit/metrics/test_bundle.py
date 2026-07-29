"""MetricBundle: config-built stage-cloned metrics, dict-compute leaf selection, directions."""

from __future__ import annotations

import torch
from torchmetrics.detection import MeanAveragePrecision

from src.core.enums import Stage
from src.metrics.bundle import DETECTION_DEFAULT_METRICS, build_metric_bundle
from src.metrics.registry import metric_factories


def _one_detection() -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
    box = torch.tensor([[10.0, 10.0, 40.0, 40.0]])
    prediction = [{"boxes": box, "scores": torch.tensor([0.9]), "labels": torch.tensor([0])}]
    target = [{"boxes": box, "labels": torch.tensor([0])}]
    return prediction, target


class TestRegistryEntry:
    def test_map_registered(self) -> None:
        assert metric_factories.get("map") is MeanAveragePrecision


class TestBuildMetricBundle:
    def test_default_spec_builds_mean_average_precision(self) -> None:
        bundle = build_metric_bundle(None, default_spec=DETECTION_DEFAULT_METRICS)
        prediction, target = _one_detection()
        bundle.update(Stage.VAL, prediction, target)
        items = bundle.log_items(Stage.VAL)
        assert set(items) == {"map50", "map50_95"}  # leaf selection, not the full mAP dict
        assert items["map50"] == torch.tensor(1.0)

    def test_custom_kwargs_reach_the_metric(self) -> None:
        bundle = build_metric_bundle({"map": {"box_format": "xyxy", "class_metrics": True}}, DETECTION_DEFAULT_METRICS)
        metric = bundle.stage_metrics(Stage.VAL)["map"]
        assert isinstance(metric, MeanAveragePrecision) and metric.class_metrics is True

    def test_stages_are_isolated_and_resettable(self) -> None:
        bundle = build_metric_bundle(None, DETECTION_DEFAULT_METRICS)
        prediction, target = _one_detection()
        bundle.update(Stage.VAL, prediction, target)
        assert bundle.stage_metrics(Stage.TEST)["map"].update_count == 0
        bundle.reset(Stage.VAL)
        assert bundle.stage_metrics(Stage.VAL)["map"].update_count == 0

    def test_directions_come_from_higher_is_better(self) -> None:
        bundle = build_metric_bundle(None, DETECTION_DEFAULT_METRICS)
        assert bundle.directions() == {"map50": True, "map50_95": True}

    def test_scalar_metric_logs_under_its_label(self) -> None:
        bundle = build_metric_bundle({"mse": None}, default_spec=DETECTION_DEFAULT_METRICS)
        bundle.update(Stage.VAL, torch.tensor([1.0]), torch.tensor([1.0]))
        assert set(bundle.log_items(Stage.VAL)) == {"mse"}
