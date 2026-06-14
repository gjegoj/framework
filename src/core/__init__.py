"""Core: framework-agnostic entities, ports, registry and runtime context.

This is the innermost circle. It depends only on ``torch`` (the numerical
language) and the standard library — never on Lightning, Hydra, data formats or
model zoos. Outer layers implement the ports defined here.
"""

from src.core.entities import (
    Batch,
    FeatureBundle,
    HeadSpec,
    LossResult,
    ModelOutput,
    Sample,
    TargetView,
    Task,
)
from src.core.enums import Stage
from src.core.ports import (
    Activation,
    Backbone,
    Criterion,
    Head,
    LossAggregator,
    MetricSet,
    TaskCodec,
)
from src.core.registry import Registry
from src.core.runtime import RuntimeContext

__all__ = [
    "Activation",
    "Backbone",
    "Batch",
    "Criterion",
    "FeatureBundle",
    "Head",
    "HeadSpec",
    "LossAggregator",
    "LossResult",
    "MetricSet",
    "ModelOutput",
    "Registry",
    "RuntimeContext",
    "Sample",
    "Stage",
    "TargetView",
    "Task",
    "TaskCodec",
]
