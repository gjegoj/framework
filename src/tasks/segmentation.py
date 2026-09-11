"""The same decisions, taken at every pixel: the topology changes, the label semantics do not."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from src.core import Axis, Stream, TargetInfo, TensorShape
from src.tasks.base import Task
from src.tasks.labels import CLASSIFICATION_METRICS, BinaryLabels, MulticlassLabels
from src.tasks.registry import task_registry

# Per-pixel per-class f1 *is* dice (2TP / (2TP + FP + FN)), so the customary score is already on this
# list under f1's name; iou adds the strict-overlap reading.
SEGMENTATION_METRICS: Mapping[str, Mapping[str, object]] = {
    "iou": {"name": "iou", "average": "none"},
    **CLASSIFICATION_METRICS,
}


class DenseOutput(Task):
    """A decision per pixel: the head reads a feature map and keeps its extent.

    A dense cell is a mask file whatever its labels mean, so the encoder and the metrics belong to the
    topology rather than to the semantics paired with it.
    """

    default_head: ClassVar[Mapping[str, object]] = {"name": "conv", "input": Stream.DECODER}
    default_target_encoder: ClassVar[str | None] = "mask"
    default_metrics: ClassVar[Mapping[str, Mapping[str, object]]] = SEGMENTATION_METRICS

    @classmethod
    def output_shape(cls, info: TargetInfo) -> TensorShape:
        """One prediction per pixel; the extent is the picture's, known only when a batch arrives."""
        return TensorShape(axes=(Axis.CLASSES, Axis.HEIGHT, Axis.WIDTH), sizes=(cls.out_features(info), None, None))


@task_registry.register("segmentation")
class Segmentation(DenseOutput, MulticlassLabels):
    """One of the declared classes per pixel."""


@task_registry.register("binary_segmentation")
class BinarySegmentation(DenseOutput, BinaryLabels):
    """One score per pixel: how much it belongs to the thing."""
