"""The framework-agnostic center: entities, ports and taxonomy, on torch and stdlib only.

## Two ways to say a capability is optional, and which to reach for

An optional capability is declared here in one of two ways, and the choice is not a
matter of taste — it follows from *who could have it*.

**A concrete default on the port**, when the capability belongs to a port the object
already implements. ``Model.task_parameters`` returns ``()``, ``Model.criterion_of``
returns ``None``, ``Backbone.native_head`` returns ``None``, ``DataModule.collate``
returns ``None``, ``DataModule.statistics`` returns an empty ``DatasetStatistics``.
Every implementation of that port answers the question, and the default is the honest
answer for the ones with nothing to add — so a caller writes one call and reads a value,
never a narrowing.

**A structural protocol**, when the capability could turn up on anything. A tracker, a
callback and a Lightning module are unrelated types, and what a consumer wants to know
is whether *this* object can draw a curve, awaits a preview, or declares which way its
metrics point. ``CurveLogger`` and its five siblings, ``MultiReadingMetric``,
``AwaitsPreview`` and ``DeclaresMetricDirections`` are that shape — and the caller
narrows with ``isinstance``, because there is no port it could have asked instead.

The line between them: **does the capability belong to a contract this object already
signed?** If yes, extend the contract with a default. If no, declare a protocol. A
fourteenth capability added by coin-flip is how a codebase ends up narrowing types that
had a perfectly good method, or bolting members onto a port that most implementations
must then refuse.

## Why three ABCs have a typed ``__call__`` and two do not

``Backbone``, ``Head`` and ``Criterion`` each override ``__call__`` with a typed
delegate, because they *are* called that way — ``backbone(inputs)``, ``head(features)``,
``criterion(logits, target)`` — and ``nn.Module.__call__`` erases the return type to
``Any`` at every one of those sites.

``Model`` and ``MetricSet`` have no such delegate, and are not missing one: nothing calls
them. A model is asked for ``step(batch)`` or ``predict(batch)``, a metric set for
``update`` / ``compute`` / ``reset``. Adding a delegate would type a call that is never
made.
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
from src.core.taxonomy import InputTopology, Modality, Objective, OutputTopology, Stage, Stream

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
