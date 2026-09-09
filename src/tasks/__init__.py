"""The tasks capability: what a task is, and the kinds a run may declare."""

from __future__ import annotations

from src.tasks.entities import Overrides, Task
from src.tasks.kinds import (
    BinaryClassification,
    BinaryLabels,
    BinarySegmentation,
    Classification,
    Contrastive,
    DenseOutput,
    Detection,
    MetricLearning,
    MulticlassLabels,
    MultilabelClassification,
    MultilabelLabels,
    MultilabelSegmentation,
    Ranking,
    Regression,
    Segmentation,
    TaskKind,
)

__all__ = [
    "BinaryClassification",
    "BinaryLabels",
    "BinarySegmentation",
    "Classification",
    "Contrastive",
    "DenseOutput",
    "Detection",
    "MetricLearning",
    "MulticlassLabels",
    "MultilabelClassification",
    "MultilabelLabels",
    "MultilabelSegmentation",
    "Overrides",
    "Ranking",
    "Regression",
    "Segmentation",
    "Task",
    "TaskKind",
]
