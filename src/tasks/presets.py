"""Task presets: the familiar, user-facing facade over ``TaskBuilder``.

A preset fixes the topology and a default objective so the simple case is one
call (``classification(name, num_classes)``), while ``objective`` stays
overridable for variants like multilabel.
"""

from __future__ import annotations

from src.core.entities import Task
from src.core.registry import Registry
from src.tasks.builder import TaskBuilder
from src.tasks.strategies.objective import objective_strategies
from src.tasks.strategies.topology import topology_strategies
from src.tasks.taxonomy import Objective, Topology

task_presets: Registry[Task] = Registry("task_preset")


@task_presets.register("classification")
def classification(name: str, num_classes: int, objective: str | None = None, weight: float = 1.0) -> Task:
    """Build a global (per-image) classification task.

    Parameters:
        name (str): Unique task name.
        num_classes (int): Number of classes.
        objective (str | None): Label semantics; defaults to ``"multiclass"``.
        weight (float): Weight in the aggregated loss.

    Returns:
        Task: The assembled classification task.
    """
    objective_kind = Objective(objective) if objective is not None else Objective.MULTICLASS
    builder = TaskBuilder(
        topology=topology_strategies.create(Topology.GLOBAL),
        objective=objective_strategies.create(objective_kind),
    )
    return builder.build(name=name, num_classes=num_classes, weight=weight)
