"""The framework-agnostic centre: entities, ports and taxonomy, on torch and stdlib only.

An optional capability is a concrete default on a port when every implementation of that
port can answer (``Model.task_parameters`` returns ``()``), and a structural protocol
(``CurveLogger``, ``AwaitsPreview``) when it could turn up on unrelated types. ``Backbone``,
``Head`` and ``Criterion`` type their ``__call__`` because they are called that way and
``nn.Module.__call__`` returns ``Any``; ``Model`` and ``MetricSet`` are never called.
"""

from __future__ import annotations

from src.core import log_keys
from src.core.choices import one_of
from src.core.entities import (
    AdaptedTarget,
    Batch,
    DataProfile,
    Features,
    Instances,
    Loss,
    Prediction,
    Sample,
    StepResult,
    TargetFacts,
    Task,
    TaskOutput,
    require_tensor,
)
from src.core.ports import (
    Activation,
    Backbone,
    Criterion,
    DataModule,
    Head,
    MetricSet,
    Model,
    MultiReadingMetric,
    SampleTransform,
    TargetAdapter,
)
from src.core.registry import Registry
from src.core.reporting import Curve, Matrix, PerClass
from src.core.taxonomy import Geometry, InputTopology, Modality, Objective, OutputTopology, Stage, Stream

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
    "Geometry",
    "Head",
    "InputTopology",
    "Instances",
    "Loss",
    "Matrix",
    "MetricSet",
    "Modality",
    "Model",
    "MultiReadingMetric",
    "Objective",
    "OutputTopology",
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
    "log_keys",
    "one_of",
    "require_tensor",
]
