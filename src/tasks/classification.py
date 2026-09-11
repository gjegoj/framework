"""Whole-sample decisions: one class, one score, or one score per label."""

from __future__ import annotations

from src.tasks.labels import BinaryLabels, MulticlassLabels, MultilabelLabels
from src.tasks.registry import task_registry


@task_registry.register("classification")
class Classification(MulticlassLabels):
    """One of the declared classes per sample."""


@task_registry.register("binary_classification")
class BinaryClassification(BinaryLabels):
    """One score per sample: how much it is the thing."""


@task_registry.register("multilabel_classification")
class MultilabelClassification(MultilabelLabels):
    """Any number of the declared labels per sample."""
