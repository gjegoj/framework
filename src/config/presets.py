"""The preset table: familiar kinds of task on the config surface."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.config.components import MetricConfig
from src.config.registry import task_preset_registry
from src.core.taxonomy import Objective, Topology


class TaskPreset(BaseModel):
    """One familiar kind of task: its point on the axes, and its customary judgment.

    Nobody writes a preset, only names one — ``preset: segmentation`` expands into axes
    and metrics while the config loads, and no preset survives past validation.

    The one home of default judgment: what a kind is usually judged by reads off this
    table alone, validated into the metric grammar the moment this module imports.
    ``metrics=None`` means the kind is not judged by per-sample metrics (metric
    learning). Never a loss — a loss default derives from one axis, so a preset has
    nothing to say about it.

    Registered rather than hard-coded, so a package adds a kind without touching this
    file::

        task_preset_registry.register_instance(
            "depth",
            TaskPreset(
                topology=Topology.DENSE,
                objective=Objective.CONTINUOUS,
                metrics={"mae": MetricConfig(name="mae")},
            ),
        )
    """

    model_config = ConfigDict(frozen=True)

    topology: Topology
    objective: Objective
    metrics: dict[str, MetricConfig] | None = None


_CLASSIFICATION_METRICS = {
    "f1": MetricConfig(name="f1", average="none"),
    "precision": MetricConfig(name="precision", average="none"),
    "recall": MetricConfig(name="recall", average="none"),
    "confusion_matrix": MetricConfig(name="confusion_matrix", normalize="true"),
}
_SEGMENTATION_METRICS = {"iou": MetricConfig(name="iou", average="none"), **_CLASSIFICATION_METRICS}
_REGRESSION_METRICS = {"mae": MetricConfig(name="mae")}
_DETECTION_METRICS = {"map": MetricConfig(name="map")}
"""What a detection run is read by. The wrapper publishes a family from one pass —
mAP@50-95 beside mAP@50 and mAP@75 — so this one entry is three numbers, not one."""

task_preset_registry.register_instance(
    "classification",
    TaskPreset(topology=Topology.GLOBAL, objective=Objective.MULTICLASS, metrics=_CLASSIFICATION_METRICS),
)
task_preset_registry.register_instance(
    "binary_classification",
    TaskPreset(topology=Topology.GLOBAL, objective=Objective.BINARY, metrics=_CLASSIFICATION_METRICS),
)
task_preset_registry.register_instance(
    "multilabel_classification",
    TaskPreset(topology=Topology.GLOBAL, objective=Objective.MULTILABEL, metrics=_CLASSIFICATION_METRICS),
)
task_preset_registry.register_instance(
    "regression",
    TaskPreset(topology=Topology.GLOBAL, objective=Objective.CONTINUOUS, metrics=_REGRESSION_METRICS),
)
task_preset_registry.register_instance(
    "metric_learning",
    TaskPreset(topology=Topology.GLOBAL, objective=Objective.METRIC),
)
task_preset_registry.register_instance(
    "segmentation",
    TaskPreset(topology=Topology.DENSE, objective=Objective.MULTICLASS, metrics=_SEGMENTATION_METRICS),
)
task_preset_registry.register_instance(
    "binary_segmentation",
    TaskPreset(topology=Topology.DENSE, objective=Objective.BINARY, metrics=_SEGMENTATION_METRICS),
)
task_preset_registry.register_instance(
    "multilabel_segmentation",
    TaskPreset(topology=Topology.DENSE, objective=Objective.MULTILABEL, metrics=_SEGMENTATION_METRICS),
)
task_preset_registry.register_instance(
    "contrastive",
    TaskPreset(topology=Topology.MULTISTREAM, objective=Objective.METRIC),
)
task_preset_registry.register_instance(
    "detection",
    # `segmentation` already names the *semantic* kind, so the instance variant lands
    # under `instance_segmentation` rather than competing for this name.
    TaskPreset(topology=Topology.INSTANCES, objective=Objective.MULTICLASS, metrics=_DETECTION_METRICS),
)

task_preset_registry.register_instance(
    "ranking",
    TaskPreset(topology=Topology.MULTIVIEW, objective=Objective.METRIC),
)


def resolve_preset(name: str) -> TaskPreset:
    """The kind of task a familiar name stands for.

    Raises:
        LookupError: For an unknown name, listing the known ones.
    """
    return task_preset_registry.create(name)
