"""Task presets: the familiar, user-facing facade over ``TaskBuilder``.

A preset is a registered singleton that fixes a *topology* and a *default
objective*.  It can be used two ways from the same object:

- **YAML path** — ``task_presets.create("classification")`` returns the
  instance; wiring calls ``.build(...)`` on it with runtime parameters.
- **Python path** — the module-level names (``classification``, ``segmentation``,
  ``regression``) *are* the preset instances.  Because ``TaskPreset`` is
  callable (``__call__`` delegates to ``build``), you can write:
  ``classification("label", num_classes=3)`` directly in tests or scripts.

Adding a new preset is one ``register_instance`` call — no wrapper function
needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
        default_encoder (str | None): ``target_encoders`` key overriding the
            objective's default (e.g. segmentation needs ``"mask"``).
            ``None`` → fall back to the objective's ``default_encoder``.
        default_metrics (MetricsSpec | None): Metric spec used when the user
            gives no ``metrics:``.  ``None`` → the objective's default.
        default_loss (str | None): ``criteria`` key used when the task config omits
            ``loss:``, overriding the objective's default.  This is where the
            *method* is pinned for objectives whose loss varies (metric learning:
            triplet vs margin-ranking vs InfoNCE).  ``None`` → objective default.
        topology_kwargs (dict[str, Any]): Extra constructor kwargs forwarded to
            ``topology_strategies.create(topology, **topology_kwargs)`` for
            parameterised topologies.  Empty dict for all fixed topologies.
    """

    topology: Topology
    default_objective: Objective
    default_encoder: str | None = None
    default_metrics: MetricsSpec | None = None
    default_loss: str | None = None
    topology_kwargs: dict[str, Any] = field(default_factory=dict)

    def resolve_objective(self, override: str | None) -> Objective:
        """Return the effective objective: ``override`` if given, else the default."""
        return Objective(override) if override is not None else self.default_objective

    def build(
        self,
        name: str,
        num_classes: int,
        objective: str | None = None,
        weight: float = 1.0,
        loss: BrickSpec | None = None,
        metrics: MetricsSpec | None = None,
        head: str | dict[str, Any] | None = None,
        feature_key: str | None = None,
    ) -> Task:
        """Assemble the task from this preset's topology and the resolved objective.

        Parameters:
            name (str): Unique task name.
            num_classes (int): Class count (or output dim for regression).
            objective (str | None): Overrides the preset's default objective.
            weight (float): Weight in the aggregated loss.
            loss (BrickSpec | None): Optional loss override.
            metrics (MetricsSpec | None): Optional metrics override.
            head: Optional head override. ``None`` → backbone native.
                ``str`` → registry key. ``dict`` → ``{kind?, _target_?, ...}``.
            feature_key (str | None): Override the backbone stream this head reads.
                ``None`` → topology default (``pooled`` / ``decoder``).

        Returns:
            Task: The assembled task.
        """
        builder = TaskBuilder(
            topology=topology_strategies.create(self.topology, **self.topology_kwargs),
            objective=objective_strategies.create(self.resolve_objective(objective)),
        )
        return builder.build(
            name=name,
            num_classes=num_classes,
            weight=weight,
            loss_spec=loss if loss is not None else self.default_loss,
            metrics_spec=metrics if metrics is not None else self.default_metrics,
            head_override=head,
            feature_key_override=feature_key,
        )

    def __call__(
        self,
        name: str,
        num_classes: int,
        objective: str | None = None,
        weight: float = 1.0,
        loss: BrickSpec | None = None,
        metrics: MetricsSpec | None = None,
        head: str | dict[str, Any] | None = None,
        feature_key: str | None = None,
    ) -> Task:
        """Shorthand for ``build`` — lets preset instances be used as functions."""
        return self.build(name, num_classes, objective, weight, loss, metrics, head, feature_key)


task_presets: Registry[TaskPreset] = Registry("task_preset")

# Per-class metrics (average="none"): torchmetrics returns a [C] tensor.
# _log_metric in LitModule logs the mean as the primary key and per-class
# values as {key}_class{i} — full per-class logging lands in M4 (typed metrics).
_PER_CLASS: dict[str, str] = {"average": "none"}

classification = TaskPreset(
    topology=Topology.GLOBAL,
    default_objective=Objective.MULTICLASS,
    default_metrics={
        "precision": _PER_CLASS,
        "recall": _PER_CLASS,
        "f1": _PER_CLASS,
        "confusion_matrix": {"normalize": "true"},
        "precision_recall_curve": None,
    },
)
task_presets.register_instance("classification", classification)

regression = TaskPreset(
    topology=Topology.GLOBAL,
    default_objective=Objective.CONTINUOUS,
    default_metrics={"mae": None},
)
task_presets.register_instance("regression", regression)

segmentation = TaskPreset(
    topology=Topology.DENSE,
    default_objective=Objective.MULTICLASS,
    default_encoder="mask",
    default_metrics={
        "iou": _PER_CLASS,
        "f1": _PER_CLASS,
        "precision": _PER_CLASS,
        "recall": _PER_CLASS,
        "confusion_matrix": {"normalize": "true"},
    },
)
task_presets.register_instance("segmentation", segmentation)

triplet = TaskPreset(
    topology=Topology.RANKING,
    default_objective=Objective.METRIC,
    default_loss="triplet_margin",  # 3 views: anchor / positive / negative
)
task_presets.register_instance("triplet", triplet)

pairwise_ranking = TaskPreset(
    topology=Topology.RANKING,
    default_objective=Objective.METRIC,
    default_loss="margin_ranking",  # 2 views ranked against each other
)
task_presets.register_instance("pairwise_ranking", pairwise_ranking)

contrastive = TaskPreset(
    topology=Topology.MULTISTREAM,
    default_objective=Objective.METRIC,
    default_loss="info_nce",  # N separate encoders aligned (InfoNCE / SigLIP)
)
task_presets.register_instance("contrastive", contrastive)
