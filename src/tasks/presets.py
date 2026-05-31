"""Task presets: the familiar, user-facing facade over ``TaskBuilder``.

A preset is one cohesive object: it fixes the *topology* and a *default
objective*, and knows how to ``build`` a task from them. The simple case is a
single call (``classification(name, num_classes)``); ``objective`` stays
overridable for variants like multilabel.

Presets are registered as singletons in ``task_presets`` so the wiring layer can
both build a task (``preset.build(...)``) and read its ``default_objective``
(to resolve the data-codec) from the *same* object — no parallel metadata table.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.entities import Task
from src.core.instantiate import BrickSpec
from src.core.registry import Registry
from src.metrics.builders import MetricsSpec
from src.tasks.builder import TaskBuilder
from src.tasks.strategies.objective import objective_strategies
from src.tasks.strategies.topology import topology_strategies
from src.tasks.taxonomy import Objective, Topology


@dataclass(frozen=True)
class TaskPreset:
    """A named (topology, default-objective) pairing that builds tasks.

    Parameters:
        topology (Topology): Output-structure axis this preset fixes.
        default_objective (Objective): Label-semantics axis used when the task
            config omits ``objective:``.
    """

    topology: Topology
    default_objective: Objective

    def resolve_objective(self, override: str | None) -> Objective:
        """Return the effective objective: the ``override`` if given, else the default."""
        return Objective(override) if override is not None else self.default_objective

    def build(
        self,
        name: str,
        num_classes: int,
        objective: str | None = None,
        weight: float = 1.0,
        loss: BrickSpec | None = None,
        metrics: MetricsSpec | None = None,
    ) -> Task:
        """Assemble the task from this preset's topology and the resolved objective.

        Parameters:
            name (str): Unique task name.
            num_classes (int): Class count (or output dim for regression).
            objective (str | None): Overrides the preset's default objective.
            weight (float): Weight in the aggregated loss.
            loss (BrickSpec | None): Optional YAML ``loss:`` override.
            metrics (MetricsSpec | None): Optional YAML ``metrics:`` override.

        Returns:
            Task: The assembled task.
        """
        builder = TaskBuilder(
            topology=topology_strategies.create(self.topology),
            objective=objective_strategies.create(self.resolve_objective(objective)),
        )
        return builder.build(name=name, num_classes=num_classes, weight=weight, loss_spec=loss, metrics_spec=metrics)


task_presets: Registry[TaskPreset] = Registry("task_preset")

task_presets.register_instance("classification", TaskPreset(Topology.GLOBAL, Objective.MULTICLASS))
task_presets.register_instance("regression", TaskPreset(Topology.GLOBAL, Objective.CONTINUOUS))


def classification(
    name: str,
    num_classes: int,
    objective: str | None = None,
    weight: float = 1.0,
    loss: BrickSpec | None = None,
    metrics: MetricsSpec | None = None,
) -> Task:
    """Build a global (per-image) classification task — facade over the preset."""
    return task_presets.create("classification").build(name, num_classes, objective, weight, loss, metrics)


def regression(
    name: str,
    num_classes: int = 1,
    objective: str | None = None,
    weight: float = 1.0,
    loss: BrickSpec | None = None,
    metrics: MetricsSpec | None = None,
) -> Task:
    """Build a global (per-image) regression task — facade over the preset.

    ``num_classes`` carries the output dimension (1 for a scalar target).
    """
    return task_presets.create("regression").build(name, num_classes, objective, weight, loss, metrics)
