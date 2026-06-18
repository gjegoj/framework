"""Annotators: turn a task's step predictions/targets into ``Label`` fields.

The model-output side of the visualization pipeline. Each annotator handles one
``(Topology, Objective)`` combination and writes ``{task}_gt`` / ``{task}_pred``
fields (plus a correctness tag) onto a ``SampleView`` for one batch element.
Selected via the ``annotators`` registry keyed by the task's composition axes —
mirroring how ``TaskBuilder`` selects bricks. New task types = new annotator (OCP).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import Tensor

from src.core.entities import Task, TaskStepView
from src.core.registry import Registry
from src.tasks.taxonomy import Objective, Topology
from src.visualization.entities import (
    Classification,
    Classifications,
    Regression,
    RegressionComponent,
    SampleView,
    Segmentation,
    SegmentationClass,
)

annotators: Registry[Annotator] = Registry("annotator")


def axes_key(topology: Topology, objective: Objective) -> str:
    """Compose the ``annotators`` registry key from a task's composition axes.

    The key is string-encoded (e.g. ``"global:multiclass"``) so the shared
    ``Registry`` stays ``str``-keyed — avoiding a ``Hashable`` widening that would
    ripple into its ``keys()`` consumers. Still "keyed by (topology, objective)".
    """
    return f"{topology.value}:{objective.value}"


class Annotator(ABC):
    """Writes one task's ground-truth/prediction labels onto a ``SampleView``."""

    @abstractmethod
    def annotate(self, sample: SampleView, task: Task, view: TaskStepView, index: int) -> None:
        """Add ``{task}_gt`` / ``{task}_pred`` fields for batch element ``index``."""


def _class_name(names: list[str], index: int) -> str:
    return names[index] if 0 <= index < len(names) else str(index)


@annotators.register(axes_key(Topology.GLOBAL, Objective.MULTICLASS))
class ClassificationAnnotator(Annotator):
    """Multiclass single-label: argmax prediction with its probability vs the target index."""

    def annotate(self, sample: SampleView, task: Task, view: TaskStepView, index: int) -> None:
        names = task.class_names or []
        probs: Tensor = view.preds[index].detach().cpu().reshape(-1).float()
        pred_index = int(torch.argmax(probs).item())
        confidence = float(probs[pred_index].item())
        gt_index = int(view.metric_target[index].detach().cpu().reshape(-1)[0].item())

        sample.fields[f"{task.name}_gt"] = Classification(label=_class_name(names, gt_index))
        sample.fields[f"{task.name}_pred"] = Classification(label=_class_name(names, pred_index), confidence=confidence)
        sample.tags.append(f"{task.name}:{'correct' if pred_index == gt_index else 'wrong'}")


@annotators.register(axes_key(Topology.GLOBAL, Objective.BINARY))
class BinaryClassificationAnnotator(Annotator):
    """Binary single-label: threshold the single positive-class probability.

    A binary head emits one sigmoid value = P(positive class, index 1), so argmax
    (used for multiclass) would always pick index 0. Threshold it instead.

    Parameters:
        threshold (float): Decision threshold on P(positive) for the predicted class.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold

    def annotate(self, sample: SampleView, task: Task, view: TaskStepView, index: int) -> None:
        names = task.class_names or []
        positive_prob = float(view.preds[index].detach().cpu().reshape(-1)[0].item())
        pred_index = 1 if positive_prob >= self._threshold else 0
        confidence = positive_prob if pred_index == 1 else 1.0 - positive_prob
        gt_index = int(view.metric_target[index].detach().cpu().reshape(-1)[0].item())

        sample.fields[f"{task.name}_gt"] = Classification(label=_class_name(names, gt_index))
        sample.fields[f"{task.name}_pred"] = Classification(label=_class_name(names, pred_index), confidence=confidence)
        sample.tags.append(f"{task.name}:{'correct' if pred_index == gt_index else 'wrong'}")


@annotators.register(axes_key(Topology.GLOBAL, Objective.MULTILABEL))
class MultilabelAnnotator(Annotator):
    """Multilabel: sigmoid scores thresholded vs a multi-hot target.

    Parameters:
        threshold (float): Minimum sigmoid score for a label to be active.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold

    def annotate(self, sample: SampleView, task: Task, view: TaskStepView, index: int) -> None:
        names = task.class_names or []
        probs: Tensor = view.preds[index].detach().cpu().reshape(-1).float()
        target: Tensor = view.metric_target[index].detach().cpu().reshape(-1).float()

        gt_indices = torch.nonzero(target > 0.5, as_tuple=False).reshape(-1).tolist()
        pred_indices = torch.nonzero(probs >= self._threshold, as_tuple=False).reshape(-1).tolist()

        sample.fields[f"{task.name}_gt"] = Classifications(
            items=[Classification(label=_class_name(names, i)) for i in gt_indices]
        )
        sample.fields[f"{task.name}_pred"] = Classifications(
            items=[Classification(label=_class_name(names, i), confidence=float(probs[i].item())) for i in pred_indices]
        )
        correct = set(gt_indices) == set(pred_indices)
        sample.tags.append(f"{task.name}:{'correct' if correct else 'wrong'}")


def _component_names(class_names: list[str] | None, dim: int) -> list[str]:
    """Per-dimension names: from ``class_names`` when they fit, else scalar/index fallback."""
    if class_names is not None and len(class_names) == dim:
        return class_names
    return [""] if dim == 1 else [f"dim{i}" for i in range(dim)]


@annotators.register(axes_key(Topology.GLOBAL, Objective.CONTINUOUS))
class RegressionAnnotator(Annotator):
    """Continuous: per-dimension value chips; pred carries the signed ``pred - gt`` error."""

    def annotate(self, sample: SampleView, task: Task, view: TaskStepView, index: int) -> None:
        preds: Tensor = view.preds[index].detach().cpu().reshape(-1).float()
        target: Tensor = view.metric_target[index].detach().cpu().reshape(-1).float()
        names = _component_names(task.class_names, preds.numel())

        sample.fields[f"{task.name}_gt"] = Regression(
            components=[RegressionComponent(names[i], float(target[i].item())) for i in range(preds.numel())]
        )
        sample.fields[f"{task.name}_pred"] = Regression(
            components=[
                RegressionComponent(names[i], float(preds[i].item()), error=float((preds[i] - target[i]).item()))
                for i in range(preds.numel())
            ]
        )
        mae = float((preds - target).abs().mean().item())
        sample.tags.append(f"{task.name}:mae={mae:.2f}")


ClassMask = tuple[int, np.ndarray]


def _iou(a: np.ndarray, b: np.ndarray) -> float | None:
    """Intersection-over-union of two boolean masks; ``None`` when their union is empty."""
    union = int((a | b).sum())
    return int((a & b).sum()) / union if union else None


def _segmentation_from_masks(
    class_masks: list[ClassMask], names: list[str] | None, ignore_index: int | None
) -> Segmentation:
    """Assemble a ``Segmentation`` from ``(class index, boolean mask)`` pairs.

    Skips ``ignore_index`` and empty masks; names classes via the ``class_names`` fallback.
    """
    classes: list[SegmentationClass] = []
    for value, mask in class_masks:
        if value == ignore_index or not mask.any():
            continue
        classes.append(SegmentationClass(_class_name(names or [], value), mask))
    return Segmentation(classes)


def _mean_iou(ious: list[float | None]) -> float:
    present = [iou for iou in ious if iou is not None]
    return float(np.mean(present)) if present else 0.0


@annotators.register(axes_key(Topology.DENSE, Objective.MULTICLASS))
class SegmentationAnnotator(Annotator):
    """Dense semantic segmentation (mutually-exclusive classes): per-class masks from the
    argmax of the prediction and the ground-truth label map.

    Parameters:
        ignore_index (int | None): A class index to skip (e.g. background); ``None`` renders
            every present class.
    """

    def __init__(self, ignore_index: int | None = None) -> None:
        self._ignore_index = ignore_index

    def annotate(self, sample: SampleView, task: Task, view: TaskStepView, index: int) -> None:
        prediction_map = view.preds[index].detach().cpu().argmax(dim=0).numpy()  # [H, W] class indices
        ground_truth_map = view.metric_target[index].detach().cpu().long().numpy()  # [H, W] class indices
        class_names = task.class_names
        ground_truth_masks: list[ClassMask] = [
            (class_index, ground_truth_map == class_index)
            for class_index in sorted(int(value) for value in np.unique(ground_truth_map))
        ]
        prediction_masks: list[ClassMask] = [
            (class_index, prediction_map == class_index)
            for class_index in sorted(int(value) for value in np.unique(prediction_map))
        ]
        sample.fields[f"{task.name}_gt"] = _segmentation_from_masks(ground_truth_masks, class_names, self._ignore_index)
        sample.fields[f"{task.name}_pred"] = _segmentation_from_masks(prediction_masks, class_names, self._ignore_index)
        present_classes = (
            {class_index for class_index, _ in ground_truth_masks}
            | {class_index for class_index, _ in prediction_masks}
        ) - {self._ignore_index}
        ious = [_iou(ground_truth_map == class_index, prediction_map == class_index) for class_index in present_classes]
        sample.tags.append(f"{task.name}:miou={_mean_iou(ious):.2f}")


@annotators.register(axes_key(Topology.DENSE, Objective.MULTILABEL))
class MultilabelSegmentationAnnotator(Annotator):
    """Dense multilabel segmentation (independent classes): per-class masks from sigmoid
    scores thresholded per channel — classes may overlap spatially.

    Parameters:
        threshold (float): Minimum sigmoid score for a pixel to belong to a class.
        ignore_index (int | None): A class index to skip; ``None`` renders every class.
    """

    def __init__(self, threshold: float = 0.5, ignore_index: int | None = None) -> None:
        self._threshold = threshold
        self._ignore_index = ignore_index

    def annotate(self, sample: SampleView, task: Task, view: TaskStepView, index: int) -> None:
        class_scores = view.preds[index].detach().cpu().numpy()  # [C, H, W] sigmoid scores
        ground_truth = view.metric_target[index].detach().cpu().numpy()  # [C, H, W] multi-hot
        class_names = task.class_names
        ground_truth_masks: list[ClassMask] = [
            (channel, ground_truth[channel] > 0.5) for channel in range(ground_truth.shape[0])
        ]
        prediction_masks: list[ClassMask] = [
            (channel, class_scores[channel] >= self._threshold) for channel in range(class_scores.shape[0])
        ]
        sample.fields[f"{task.name}_gt"] = _segmentation_from_masks(ground_truth_masks, class_names, self._ignore_index)
        sample.fields[f"{task.name}_pred"] = _segmentation_from_masks(prediction_masks, class_names, self._ignore_index)
        ious = [
            _iou(ground_truth_mask, prediction_mask)
            for (channel, ground_truth_mask), (_, prediction_mask) in zip(ground_truth_masks, prediction_masks)
            if channel != self._ignore_index
        ]
        sample.tags.append(f"{task.name}:miou={_mean_iou(ious):.2f}")
