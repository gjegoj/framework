"""Core: framework-agnostic entities, ports, registry and runtime context.

This is the innermost circle. It depends only on ``torch`` (the numerical
language) and the standard library — never on Lightning, Hydra, data formats or
model zoos. Outer layers implement the ports defined here.

Re-exports the full domain surface — every entity and port — so ``from src.core
import X`` is consistent. The string-key/constant/instantiate utilities keep their
own convention (``from src.core.keys import IMAGE``, etc.).
"""

from src.core.entities import (
    Batch,
    BatchMeta,
    FeatureBundle,
    HeadSpec,
    LossResult,
    ModelOutput,
    Sample,
    SampleMeta,
    StepOutput,
    TargetView,
    Task,
    TaskStepView,
    is_step_output,
)
from src.core.enums import Stage
from src.core.ports import (
    Activation,
    Backbone,
    Criterion,
    Head,
    LossAggregator,
    MetricDirectionProvider,
    MetricSet,
    PlotLogger,
    TargetAdapter,
)
from src.core.registry import Registry
from src.core.runtime import RuntimeContext

__all__ = [
    "Activation",
    "Backbone",
    "Batch",
    "BatchMeta",
    "Criterion",
    "FeatureBundle",
    "Head",
    "HeadSpec",
    "LossAggregator",
    "LossResult",
    "MetricDirectionProvider",
    "MetricSet",
    "ModelOutput",
    "PlotLogger",
    "Registry",
    "RuntimeContext",
    "Sample",
    "SampleMeta",
    "Stage",
    "StepOutput",
    "TargetAdapter",
    "TargetView",
    "Task",
    "TaskStepView",
    "is_step_output",
]
