"""Annotators: turn a task's step predictions/targets into ``Label`` fields.

The model-output side of the visualization pipeline. Each annotator handles one
``(Topology, Objective)`` combination and writes ``{task}_gt`` / ``{task}_pred``
fields (plus a correctness tag) onto a ``SampleView`` for one batch element.
Selected via the ``annotators`` registry keyed by the task's composition axes —
mirroring how ``TaskBuilder`` selects bricks. New task types = new annotator (OCP).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from src.core.entities import Task, TaskStepView
from src.core.registry import Registry
from src.tasks.taxonomy import Objective, Topology
from src.visualization.entities import Classification, Classifications, SampleView

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
@annotators.register(axes_key(Topology.GLOBAL, Objective.BINARY))
class ClassificationAnnotator(Annotator):
    """Single-label: argmax prediction with its probability vs the target index."""

    def annotate(self, sample: SampleView, task: Task, view: TaskStepView, index: int) -> None:
        names = task.class_names or []
        probs: Tensor = view.preds[index].detach().cpu().reshape(-1).float()
        pred_index = int(torch.argmax(probs).item())
        confidence = float(probs[pred_index].item())
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
