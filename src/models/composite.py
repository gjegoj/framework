"""The composite model family: one shared backbone, per-task heads and criteria."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, override

from torch import nn

from src.core.entities import AdaptedTarget, Loss, Prediction, StepResult, TaskOutput, as_tensor
from src.core.ports import Model
from src.core.taxonomy import Stream

if TYPE_CHECKING:
    from torch import Tensor

    from src.core.entities import Batch
    from src.core.ports import Activation, Backbone, Criterion, Head, TargetAdapter


@dataclass(frozen=True, slots=True, eq=False)
class TaskComponents:
    """How the composite family serves one ``Task``: the bricks behind its predictions.

    ``Task`` (core) *declares* a learned objective in family-agnostic terms;
    ``TaskComponents`` *materializes* that declaration for the composite
    model — the concrete head, criterion, activation, and target adapter that
    ``build_task_components`` derives from the task, data facts, and backbone
    dimensions. ``weight`` is copied in; ``Task.weight`` stays the source of
    truth.
    """

    head: Head
    criterion: Criterion
    activation: Activation
    target_adapter: TargetAdapter | None
    stream: str = Stream.FEATURES
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError(f"Task weight must be positive, got {self.weight}.")


class CompositeModel(Model):
    """Backbone x heads: encode once, serve every task from named streams.

    Heads and criteria register as submodules (``heads.<task>``,
    ``criteria.<task>``), so parametric criteria (ArcFace-style) are
    optimized and checkpointed together with the model.
    """

    BACKBONE: ClassVar[str] = "backbone"
    """The attribute the shared backbone sits under, and so the tail of the dot-path naming it.

    Published rather than spelled out where it is read: a freeze callback names
    this module in config, and assembly builds that path by joining the segment
    each owner publishes. A rename here reaches the config addressing it.
    """

    def __init__(self, backbone: Backbone, components: Mapping[str, TaskComponents]) -> None:
        super().__init__()
        if not components:
            raise ValueError("CompositeModel needs at least one task component.")
        self.backbone = backbone
        self.heads = nn.ModuleDict({name: component.head for name, component in components.items()})
        self.criteria = nn.ModuleDict({name: component.criterion for name, component in components.items()})
        self._components = dict(components)

    @override
    def step(self, batch: Batch) -> StepResult:
        features = self.backbone(batch.inputs)
        outputs: dict[str, TaskOutput] = {}
        raw: dict[str, Tensor] = {}
        metric_targets: dict[str, TaskOutput] = {}
        losses: list[Loss] = []
        for name, component in self._components.items():
            logits = component.head(features[component.stream])
            adapted = self._adapt_target(batch, name, component)
            task_loss = component.criterion(logits, adapted.for_loss).scoped(name)
            losses.append(component.weight * task_loss)
            outputs[name] = component.activation(logits)
            raw[name] = logits
            metric_targets[name] = adapted.for_metrics
        return StepResult(
            loss=Loss.sum(losses),
            prediction=Prediction(outputs=outputs, features=features, logits=raw),
            targets=metric_targets,
        )

    @override
    def predict(self, batch: Batch) -> Prediction:
        features = self.backbone(batch.inputs)
        raw = {name: component.head(features[component.stream]) for name, component in self._components.items()}
        outputs: dict[str, TaskOutput] = {
            name: self._components[name].activation(logits) for name, logits in raw.items()
        }
        return Prediction(outputs=outputs, features=features, logits=raw)

    def _adapt_target(self, batch: Batch, task_name: str, component: TaskComponents) -> AdaptedTarget:
        """Look up and shape the task's target; raw when there is nothing to shape.

        No adapter means the objective has nothing to *shape* — not necessarily
        nothing to deliver: a ranking task's per-pair preference arrives as the
        number it already is. ``absent`` is only for a target that truly is —
        structure-supervised tasks whose batch carries no column.
        """
        if component.target_adapter is None:
            declared = batch.targets.get(task_name)
            if declared is None:
                return AdaptedTarget.absent()
            raw = as_tensor(declared, task=task_name, wanted_by="a composed model")
            return AdaptedTarget(for_loss=raw, for_metrics=raw)
        return component.target_adapter(self._target(batch, task_name))

    @property
    @override
    def architecture(self) -> str:
        """The backbone's: what a composed run is, is what encodes for it."""
        return self.backbone.architecture

    @override
    def task_parameters(self, task_name: str) -> Iterable[nn.Parameter]:
        """The task's head and criterion parameters — the bricks a per-task rate moves."""
        for owner in (self.heads, self.criteria):
            if task_name in owner:
                yield from owner[task_name].parameters()

    @staticmethod
    def _target(batch: Batch, task_name: str) -> Tensor:
        try:
            return as_tensor(batch.targets[task_name], task=task_name, wanted_by="a composed model")
        except KeyError:
            available = ", ".join(sorted(batch.targets)) or "none"
            raise LookupError(f"Batch has no target for task '{task_name}'. Available targets: {available}.") from None
