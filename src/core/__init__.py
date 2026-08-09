"""The framework-agnostic center: entities, ports and taxonomy, on torch and stdlib only."""

from __future__ import annotations

from src.core import log_keys
from src.core.choices import one_of
from src.core.entities import (
    AdaptedTarget,
    Batch,
    Curve,
    DataProfile,
    Features,
    Instances,
    Loss,
    Matrix,
    PerClass,
    Prediction,
    Sample,
    StepResult,
    TargetFacts,
    Task,
    TaskOutput,
    as_tensor,
)
from src.core.ports import (
    Activation,
    Backbone,
    Criterion,
    DataModule,
    Head,
    MetricFamily,
    MetricSet,
    Model,
    SampleTransform,
    TargetAdapter,
)
from src.core.registry import Registry
from src.core.taxonomy import Modality, Objective, Stage, Stream, Topology

__all__ = [
    "Activation",
    "AdaptedTarget",
    "Backbone",
    "Batch",
    "Criterion",
    "Curve",
    "DataModule",
    "DataProfile",
    "Features",
    "Head",
    "Instances",
    "Loss",
    "Matrix",
    "MetricFamily",
    "MetricSet",
    "Modality",
    "Model",
    "Objective",
    "PerClass",
    "Prediction",
    "Registry",
    "Sample",
    "SampleTransform",
    "Stage",
    "StepResult",
    "Stream",
    "TargetAdapter",
    "TargetFacts",
    "Task",
    "TaskOutput",
    "Topology",
    "as_tensor",
    "log_keys",
    "one_of",
]
