"""Turning task declarations into entities and the bricks that serve them."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any

from src.assembly.instantiate import instantiate
from src.assembly.metrics import build_metric_sets
from src.config.tasks import LossConfig
from src.core.entities import Task
from src.losses import WeightedSumCriterion
from src.losses.registry import criterion_registry
from src.models.registry import head_registry
from src.tasks import build_task_components

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.config import ExperimentConfig
    from src.config.tasks import HeadConfig
    from src.core.entities import DataProfile, TargetFacts
    from src.core.ports import Backbone, Criterion, Head
    from src.models import TaskComponents


def build_task_entities(config: ExperimentConfig, profile: DataProfile) -> list[Task]:
    """The declared tasks as entities, with no bricks behind them.

    Split from ``build_tasks`` because a vendor family needs exactly this half: it binds
    its own head and criterion, which ``Task``'s own docstring already anticipates. Runs
    after ``DataModule.setup`` either way — metric sizes come from the profiled facts.
    """
    return [
        Task(
            name=name,
            topology=declared.topology,
            objective=declared.objective,
            metrics=build_metric_sets(declared.objective, facts=profile.facts(name), metrics=declared.metrics),
            weight=declared.weight,
            lr=declared.lr,
            class_names=profile.facts(name).class_names,
        )
        for name, declared in config.tasks.items()
    ]


def build_tasks(
    config: ExperimentConfig, profile: DataProfile, backbone: Backbone
) -> tuple[list[Task], dict[str, TaskComponents]]:
    """Build the declared tasks and the composite bricks serving each.

    Runs after ``DataModule.setup``: metric sizes and head widths come from the
    profiled facts, never from config. A task may override the objective's
    default criterion; everything else follows from its two axes.
    """
    tasks = build_task_entities(config, profile)
    components: dict[str, TaskComponents] = {}
    for task in tasks:
        name = task.name
        declared = config.tasks[name]
        facts = profile.facts(name)
        built = build_task_components(
            task,
            profile,
            backbone,
            stream=declared.stream,
            prefer_native_head=declared.native_head,
            head_factory=partial(_build_head, declared.head) if declared.head is not None else None,
        )
        if declared.loss is not None:
            criterion = _task_criterion(declared.loss, facts, embedding_dim=backbone.feature_dim(built.stream))
            built = replace(built, criterion=criterion)
        components[name] = built
    return tasks, components


def _build_head(declared: HeadConfig, in_features: int, out_features: int) -> Head:
    """The declared kind of head, at the sizes the builder resolved.

    The sizes arrive as derived values: a head receives the ones it names, so
    ``{name: cosine}`` is a complete declaration and config never repeats what
    the backbone and the data already said.
    """
    built: Head = instantiate(declared, head_registry, in_features=in_features, out_features=out_features)
    return built


def build_criterion(declared: LossConfig | Sequence[LossConfig], /, **derived: Any) -> Criterion:
    """One declared criterion, or several added with their weights.

    The weight rides on the declaration rather than beside it, so a term keeps it
    wherever the term is used — inside a task's objective, or inside the soft
    comparison a distilled run adds to that objective. A single unweighted part is
    itself, so the common case carries no wrapper.

    ``derived`` is offered to every part and reaches only the ones that name it.
    """
    parts = [declared] if isinstance(declared, LossConfig) else list(declared)
    built: list[tuple[Criterion, float]] = [
        (instantiate(part, criterion_registry, **derived), part.weight) for part in parts
    ]
    if len(built) == 1 and built[0][1] == 1.0:
        return built[0][0]
    return WeightedSumCriterion(built)


def _task_criterion(declared: LossConfig | Sequence[LossConfig], facts: TargetFacts, embedding_dim: int) -> Criterion:
    """A task's criterion, with the facts only assembly holds offered to every part.

    An expectation term gets its bin centres from the encoder that laid them out,
    a proxy criterion sizes its prototypes from the fitted vocabulary and the
    stream the task reads — and nobody pastes either into config a second time.
    """
    derived: dict[str, Any] = {"embedding_dim": embedding_dim}
    if facts.class_values is not None:
        derived["class_values"] = facts.class_values
    if facts.num_classes is not None:
        derived["num_classes"] = facts.num_classes
    return build_criterion(declared, **derived)
