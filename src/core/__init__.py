"""The framework-agnostic centre: entities, ports and taxonomy, on torch and stdlib only.

What every capability package builds on and none may reshape: the values a run is made of
(``Sample``, ``Batch``, ``Instances``, ``Features``, ``Loss``, ``Prediction``, ``StepResult``,
the facts the data revealed), the ports a model and a backbone answer to, the closed
vocabularies (``Stage``, ``Stream``, ``Geometry``, ``Modality``, ``OutputTopology``), the
log-key grammar and the registry every package catalogues its names in. ``Backbone`` and
``Criterion`` type their ``__call__`` because they are called that way and
``nn.Module.__call__`` returns ``Any``; a head is any ``nn.Module``.
"""

from __future__ import annotations

from src.core import log_keys
from src.core.choices import one_of
from src.core.entities import (
    Batch,
    DatasetFacts,
    Features,
    Instances,
    Loss,
    Prediction,
    Sample,
    StepResult,
    TaskFacts,
    TaskOutput,
    require_tensor,
)
from src.core.ports import (
    Backbone,
    Criterion,
    GeometryAware,
    Model,
    SampleTransform,
)
from src.core.registry import Registry
from src.core.taxonomy import Geometry, Modality, OutputTopology, Stage, Stream

__all__ = [
    "Backbone",
    "Batch",
    "Criterion",
    "DatasetFacts",
    "Features",
    "Geometry",
    "GeometryAware",
    "Instances",
    "Loss",
    "Modality",
    "Model",
    "OutputTopology",
    "Prediction",
    "Registry",
    "Sample",
    "SampleTransform",
    "Stage",
    "StepResult",
    "Stream",
    "TaskFacts",
    "TaskOutput",
    "log_keys",
    "one_of",
    "require_tensor",
]
