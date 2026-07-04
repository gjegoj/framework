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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.core.entities import Task
from src.core.instantiate import BrickSpec
from src.core.ports import Activation
from src.core.registry import Registry
from src.core.taxonomy import Objective, Topology
from src.metrics.builders import MetricsSpec
from src.tasks.activations import NormalizeActivation
from src.tasks.builder import TaskBuilder
from src.tasks.strategies.objective import objective_strategies
from src.tasks.strategies.topology import topology_strategies


@dataclass(frozen=True, slots=True)
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
        default_head (str | dict[str, Any] | None): Head override used when the task
            config omits ``head:`` (e.g. ArcFace's ``"cosine"`` head).  ``None`` →
            backbone native. The task config's own ``head:`` always wins over this.
        default_activation (Callable[[], Activation] | None): Activation factory used
            when the objective's own activation isn't the right fit (e.g. embedding
            presets need L2-normalized output instead of the objective default).
            ``None`` → the objective's ``build_activation()``.
        topology_kwargs (dict[str, Any]): Extra constructor kwargs forwarded to
            ``topology_strategies.create(topology, **topology_kwargs)`` for
            parameterised topologies.  Empty dict for all fixed topologies.
    """

    topology: Topology
    default_objective: Objective
    default_encoder: str | None = None
    default_metrics: MetricsSpec | None = None
    default_loss: str | None = None
    default_head: str | dict[str, Any] | None = None
    default_activation: Callable[[], Activation] | None = None
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
        class_count: int | None = None,
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
            class_count (int | None): Runtime-inferred label-vocabulary size for
                dimension-aware criteria; forwarded to ``TaskBuilder.build``.

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
            head_override=head if head is not None else self.default_head,
            feature_key_override=feature_key,
            class_count=class_count,
            activation_factory=self.default_activation,
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
        class_count: int | None = None,
    ) -> Task:
        """Shorthand for ``build`` — lets preset instances be used as functions."""
        return self.build(name, num_classes, objective, weight, loss, metrics, head, feature_key, class_count)


task_presets: Registry[TaskPreset] = Registry("task_preset")

# Per-class metrics (average="none"): torchmetrics returns a [C] tensor, which the
# typed VectorMetricHandler logs as the mean at {key}/mean plus one scalar per class
# at {key}/<class_name> (falling back to class{i}).
_PER_CLASS: dict[str, str] = {"average": "none"}

_CLASSIFICATION_METRICS: MetricsSpec = {
    "precision": _PER_CLASS,
    "recall": _PER_CLASS,
    "f1": _PER_CLASS,
    "confusion_matrix": {"normalize": "true"},
    "precision_recall_curve": None,
}

classification = TaskPreset(
    topology=Topology.GLOBAL,
    default_objective=Objective.MULTICLASS,
    default_metrics=_CLASSIFICATION_METRICS,
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
    topology=Topology.MULTIVIEW,
    default_objective=Objective.METRIC,
    default_loss="triplet_margin",  # 3 views: anchor / positive / negative
)
task_presets.register_instance("triplet", triplet)

pairwise_ranking = TaskPreset(
    topology=Topology.MULTIVIEW,
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

arcface = TaskPreset(
    topology=Topology.GLOBAL,
    default_objective=Objective.MULTICLASS,
    default_loss="arcface",
    default_head="cosine",
    default_metrics=_CLASSIFICATION_METRICS,
)
task_presets.register_instance("arcface", arcface)

arcface_embedding = TaskPreset(
    topology=Topology.GLOBAL,
    default_objective=Objective.METRIC,
    default_loss="arcface_proxy",
    default_encoder="label",
    default_head="linear",
    default_activation=NormalizeActivation,
)
task_presets.register_instance("arcface_embedding", arcface_embedding)
