"""Builds composite-family components from universal task declarations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.ports import Head
from src.models import TaskComponents, WrappedHead
from src.tasks.registry import objective_registry, topology_registry

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.core.entities import DataProfile, Task
    from src.core.ports import Backbone


def build_task_components(
    task: Task,
    profile: DataProfile,
    backbone: Backbone,
    stream: str | None = None,
    prefer_native_head: bool = False,
    head_factory: Callable[[int, int], Head] | None = None,
) -> TaskComponents:
    """Assemble the bricks that serve ``task`` inside a composite model.

    Resolves the task's axes to their behaviours, validates the pairing,
    reads inferred facts (``num_classes``) from ``profile`` when the objective
    needs them, and sizes the head from the backbone's stream dimension —
    the ordering contract that keeps output sizes out of config.

    Parameters:
        stream (str | None): Read this stream instead of the topology's
            default — e.g. a GLOBAL task on ``Stream.ENCODER`` of an smp
            backbone.
        prefer_native_head (bool): Use the backbone's own head for the
            stream (smp's segmentation/classification head, timm's
            classifier) instead of building a framework head.
        head_factory (Callable | None): Build this head instead of the
            topology's default, given ``(in_features, out_features)`` — how a
            config override arrives without config entering this layer. A
            factory rather than an instance, because the sizes are resolved
            here.

    Raises:
        LookupError: If a native head is preferred but the backbone offers
            none for the stream.
    """
    objective = objective_registry.create(task.objective)
    topology = topology_registry.create(task.topology)
    if not topology.supports(task.objective):
        raise ValueError(f"Topology '{task.topology}' cannot be supervised by objective '{task.objective}'.")
    if objective.needs_num_classes:
        profile.require_num_classes(task.name)
    facts = profile.facts(task.name)
    chosen_stream = stream if stream is not None else topology.stream
    in_features = backbone.feature_dim(chosen_stream)
    out_features = objective.out_features(facts)
    head: Head
    if head_factory is not None:
        head = head_factory(in_features, out_features)
    elif prefer_native_head:
        native = backbone.native_head(chosen_stream, in_features, out_features)
        if native is None:
            raise LookupError(f"{type(backbone).__name__} offers no native head for stream '{chosen_stream}'.")
        # A native module that already is a Head keeps its own shape: wrapping it
        # would bury contract paths (freeze's `...heads.<task>.base`) under a
        # private attribute.
        head = native if isinstance(native, Head) else WrappedHead(native)
    else:
        head = topology.build_head(in_features=in_features, out_features=out_features)
    return TaskComponents(
        head=head,
        criterion=objective.build_criterion(facts),
        activation=objective.build_activation(facts),
        target_adapter=objective.build_target_adapter(facts),
        stream=chosen_stream,
        weight=task.weight,
    )
