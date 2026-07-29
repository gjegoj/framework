"""Stage-cloned metric bundle for complete-model runs (built from the ``metrics:`` block).

The standard contour adapts torchmetrics through ``MetricSet`` (tensor logits vs
targets). Complete models produce family-specific prediction/target forms (e.g.
detection's list-of-box-dicts), so their metrics live in this bundle instead: same
config grammar and registry, per-evaluation-stage clones, log leaves and directions
derived from the metrics themselves.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

from torch import Tensor, nn
from torchmetrics import Metric
from torchmetrics.detection import MeanAveragePrecision

from src.core.enums import Stage
from src.metrics.builders import MetricsSpec, build_metric_instances

_EVALUATION_STAGES = (Stage.VAL, Stage.TEST)

DETECTION_DEFAULT_METRICS: MetricsSpec = {"map": {"box_format": "xyxy"}}

# Metrics whose compute() returns a dict expose only these leaves, renamed for the
# task/metric/stage log grammar. Everything else logs its scalar under the config label.
_DICT_COMPUTE_LEAVES: dict[type[Metric], dict[str, str]] = {
    MeanAveragePrecision: {"map_50": "map50", "map": "map50_95"},
}


class MetricBundle[PredictionT, TargetT](nn.Module):
    """Per-stage metric clones with a uniform update/log/reset surface.

    Generic over the prediction/target pair so pairing with a
    ``CompleteModel[PredictionT, TargetT]`` is checked statically.

    Parameters:
        metrics (dict[str, Metric]): Label -> metric prototype (cloned per stage).
    """

    def __init__(self, metrics: dict[str, Metric]) -> None:
        super().__init__()
        self._stage_metrics = nn.ModuleDict(
            {
                stage: nn.ModuleDict({label: deepcopy(metric) for label, metric in metrics.items()})
                for stage in _EVALUATION_STAGES
            }
        )

    def stage_metrics(self, stage: Stage) -> dict[str, Metric]:
        """The stage's metric instances by label (ModuleDict lookup erases types).

        Parameters:
            stage (Stage): Evaluation stage (``VAL`` or ``TEST``).

        Returns:
            dict[str, Metric]: Label -> the stage's own metric instance.
        """
        module_dict = cast("nn.ModuleDict", self._stage_metrics[stage])
        return cast("dict[str, Metric]", dict(module_dict.items()))

    def update(self, stage: Stage, predictions: PredictionT, targets: TargetT) -> None:
        """Accumulate one batch of decoded predictions against ground truth.

        Parameters:
            stage (Stage): Evaluation stage whose metric state to update.
            predictions (PredictionT): Metric-ready predictions from the model.
            targets (TargetT): Metric-ready ground truth from the batch.
        """
        for metric in self.stage_metrics(stage).values():
            metric.update(predictions, targets)

    def log_items(self, stage: Stage) -> dict[str, Tensor]:
        """Computed scalar leaves for logging: ``{leaf: value}`` (un-namespaced).

        Parameters:
            stage (Stage): Evaluation stage to compute.

        Returns:
            dict[str, Tensor]: Leaf key -> scalar value.
        """
        items: dict[str, Tensor] = {}
        for label, metric in self.stage_metrics(stage).items():
            value = metric.compute()
            leaves = _DICT_COMPUTE_LEAVES.get(type(metric))
            if leaves is not None:
                items.update({leaf: value[source] for source, leaf in leaves.items()})
            else:
                items[label] = value
        return items

    def directions(self) -> dict[str, bool | None]:
        """Leaf -> higher-is-better, read from each metric's own declaration."""
        directions: dict[str, bool | None] = {}
        for label, metric in self.stage_metrics(_EVALUATION_STAGES[0]).items():
            leaves = _DICT_COMPUTE_LEAVES.get(type(metric), {label: label}).values()
            directions.update(dict.fromkeys(leaves, metric.higher_is_better))
        return directions

    def reset(self, stage: Stage) -> None:
        """Clear the stage's accumulated metric state (after epoch-end logging).

        Parameters:
            stage (Stage): Evaluation stage to reset.
        """
        for metric in self.stage_metrics(stage).values():
            metric.reset()


def build_metric_bundle(spec: MetricsSpec | None, default_spec: MetricsSpec) -> MetricBundle[Any, Any]:
    """Build a bundle from a task's ``metrics:`` block (or the family default).

    Parameters:
        spec (MetricsSpec | None): The task's ``metrics:`` block; ``None`` -> default.
        default_spec (MetricsSpec): Family default, e.g. ``DETECTION_DEFAULT_METRICS``.

    Returns:
        MetricBundle[Any, Any]: Fresh bundle (typed by the caller's pairing).
    """
    instances = build_metric_instances(spec, base_kwargs={}, default_spec=default_spec)
    metrics = {label: metric for label, metric in instances.items() if isinstance(metric, Metric)}
    return MetricBundle(metrics)
