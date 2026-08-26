"""The tasks capability: from universal ``Task`` declarations to the composite family's components."""

from __future__ import annotations

from src.tasks.builder import build_task_components, default_target_encoder
from src.tasks.objectives import (
    BinaryObjective,
    ContinuousObjective,
    MetricObjective,
    MulticlassObjective,
    MultilabelObjective,
    TaskObjective,
)
from src.tasks.topologies import (
    DenseTopology,
    GlobalTopology,
    InstancesTopology,
    TaskTopology,
)

__all__ = [
    "BinaryObjective",
    "ContinuousObjective",
    "DenseTopology",
    "GlobalTopology",
    "InstancesTopology",
    "MetricObjective",
    "MulticlassObjective",
    "MultilabelObjective",
    "TaskObjective",
    "TaskTopology",
    "build_task_components",
    "default_target_encoder",
]
