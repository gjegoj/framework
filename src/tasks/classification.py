"""Whole-sample decisions: one class, one score, or one score per label.

A kind is a topology paired with label semantics, and the whole-sample topology is the one ``Task``
already describes — a pooled feature, one prediction per sample — so each kind here is a semantics from
``semantics.py`` under the name a run writes for it. ``segmentation.py`` shows the same pairing where the
topology has something of its own to say.
"""

from __future__ import annotations

from src.tasks.registry import task_registry
from src.tasks.semantics import BinarySemantics, MulticlassSemantics, MultilabelSemantics


@task_registry.register("classification")
class Classification(MulticlassSemantics):
    """One of the declared classes per sample."""


@task_registry.register("binary_classification")
class BinaryClassification(BinarySemantics):
    """One score per sample: how much it is the thing."""


@task_registry.register("multilabel_classification")
class MultilabelClassification(MultilabelSemantics):
    """Any number of the declared labels per sample."""
