"""The preset table: familiar kinds of task on the config surface."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from src.config.components import MetricConfig
from src.config.registry import task_preset_registry
from src.core.taxonomy import InputTopology, Objective, OutputTopology


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
    file — a decorated subclass whose defaults *are* the point, built and validated
    the moment the module imports::

        @task_preset_registry.register_instance("depth")
        class Depth(TaskPreset):
            output_topology: OutputTopology = OutputTopology.DENSE
            objective: Objective = Objective.CONTINUOUS
            metrics: dict[str, MetricConfig] | None = {"mae": MetricConfig(name="mae")}
    """

    # A preset is declared far from any config validator, so its declaration must go
    # through the grammar *at declaration time*, in both spellings. ``extra="forbid"``
    # turns a typo'd argument into an error naming it; ``validate_default=True`` puts a
    # subclass's field defaults — the decorator idiom's way of declaring — through the
    # same validation (measured: without it a malformed metrics default was accepted
    # and served as a raw dict, because pydantic validates calls, not class defaults).
    # The decorator constructs eagerly, so both fire at import.
    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Refuse a subclass declaring fields ``TaskPreset`` does not have.

        The subclass-form analogue of ``extra="forbid"``, which sees extra *arguments*
        but not extra *fields*: without this, a typo'd field name becomes a new model
        field instead of an error naming it. Runs during class creation, after pydantic
        collects the fields — the one window where the comparison is possible.
        """
        super().__pydantic_init_subclass__(**kwargs)
        invented = set(cls.model_fields) - set(TaskPreset.model_fields)
        if invented:
            raise TypeError(
                f"{cls.__name__} declares unknown field(s) {sorted(invented)}; a preset gives "
                f"defaults for TaskPreset's own fields: {sorted(TaskPreset.model_fields)}."
            )

    output_topology: OutputTopology
    objective: Objective
    input_topology: InputTopology = InputTopology.SINGLE
    metrics: dict[str, MetricConfig] | None = None


_CLASSIFICATION_METRICS = {
    # `average="none"` asks for the per-class vector; torchmetrics' binary task
    # ignores it and answers with the positive class's scalar (measured), so one
    # dict serves all three classification objectives without lying on any.
    "f1": MetricConfig(name="f1", average="none"),
    "precision": MetricConfig(name="precision", average="none"),
    "recall": MetricConfig(name="recall", average="none"),
    "confusion_matrix": MetricConfig(name="confusion_matrix", normalize="true"),
}
# Per-pixel per-class f1 IS dice (2TP/(2TP+FP+FN)), so the customary dice score
# is already on this list under f1's name; iou adds the strict-overlap reading.
_SEGMENTATION_METRICS = {"iou": MetricConfig(name="iou", average="none"), **_CLASSIFICATION_METRICS}
_REGRESSION_METRICS = {"mae": MetricConfig(name="mae")}
_DETECTION_METRICS = {"map": MetricConfig(name="map")}
"""What a detection run is read by. The wrapper publishes a family from one pass —
mAP@50-95 beside mAP@50 and mAP@75 — so this one entry is three numbers, not one."""


@task_preset_registry.register_instance("classification")
class Classification(TaskPreset):
    output_topology: OutputTopology = OutputTopology.GLOBAL
    objective: Objective = Objective.MULTICLASS
    metrics: dict[str, MetricConfig] | None = _CLASSIFICATION_METRICS


@task_preset_registry.register_instance("binary_classification")
class BinaryClassification(TaskPreset):
    output_topology: OutputTopology = OutputTopology.GLOBAL
    objective: Objective = Objective.BINARY
    metrics: dict[str, MetricConfig] | None = _CLASSIFICATION_METRICS


@task_preset_registry.register_instance("multilabel_classification")
class MultilabelClassification(TaskPreset):
    output_topology: OutputTopology = OutputTopology.GLOBAL
    objective: Objective = Objective.MULTILABEL
    metrics: dict[str, MetricConfig] | None = _CLASSIFICATION_METRICS


@task_preset_registry.register_instance("regression")
class Regression(TaskPreset):
    output_topology: OutputTopology = OutputTopology.GLOBAL
    objective: Objective = Objective.CONTINUOUS
    metrics: dict[str, MetricConfig] | None = _REGRESSION_METRICS


@task_preset_registry.register_instance("metric_learning")
class MetricLearning(TaskPreset):
    output_topology: OutputTopology = OutputTopology.GLOBAL
    objective: Objective = Objective.METRIC


@task_preset_registry.register_instance("segmentation")
class Segmentation(TaskPreset):
    output_topology: OutputTopology = OutputTopology.DENSE
    objective: Objective = Objective.MULTICLASS
    metrics: dict[str, MetricConfig] | None = _SEGMENTATION_METRICS


@task_preset_registry.register_instance("binary_segmentation")
class BinarySegmentation(TaskPreset):
    output_topology: OutputTopology = OutputTopology.DENSE
    objective: Objective = Objective.BINARY
    metrics: dict[str, MetricConfig] | None = _SEGMENTATION_METRICS


@task_preset_registry.register_instance("multilabel_segmentation")
class MultilabelSegmentation(TaskPreset):
    output_topology: OutputTopology = OutputTopology.DENSE
    objective: Objective = Objective.MULTILABEL
    metrics: dict[str, MetricConfig] | None = _SEGMENTATION_METRICS


@task_preset_registry.register_instance("contrastive")
class Contrastive(TaskPreset):
    output_topology: OutputTopology = OutputTopology.GLOBAL
    input_topology: InputTopology = InputTopology.MULTISTREAM
    objective: Objective = Objective.METRIC


@task_preset_registry.register_instance("detection")
class Detection(TaskPreset):
    """``segmentation`` already names the *semantic* kind, so the instance variant
    lands under ``instance_segmentation`` rather than competing for this name."""

    output_topology: OutputTopology = OutputTopology.INSTANCES
    objective: Objective = Objective.MULTICLASS
    metrics: dict[str, MetricConfig] | None = _DETECTION_METRICS


@task_preset_registry.register_instance("ranking")
class Ranking(TaskPreset):
    output_topology: OutputTopology = OutputTopology.GLOBAL
    input_topology: InputTopology = InputTopology.MULTIVIEW
    objective: Objective = Objective.METRIC


def resolve_preset(name: str) -> TaskPreset:
    """The kind of task a familiar name stands for.

    Raises:
        LookupError: For an unknown name, listing the known ones.
    """
    return task_preset_registry.create(name)
