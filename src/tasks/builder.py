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
    topology = topology_registry.create(task.output_topology)
    # Both halves of "can this framework serve this task?", asked together and before
    # anything is built. The second used to be a refusal thrown from inside `build_head`,
    # i.e. from a method the builder was never meant to reach for such a task.
    if not topology.supports(task.objective, task.input_topology):
        raise ValueError(
            f"Output topology '{task.output_topology}' with input topology '{task.input_topology}' "
            f"cannot be supervised by objective '{task.objective}'."
        )
    if not topology.composes_head:
        raise ValueError(
            f"Task '{task.name}' is '{task.output_topology}', whose head belongs to the model family that "
            f"owns it — its assigner and its loss are part of the same design, and this framework "
            f"composes none of them. Declare a vendor family instead, e.g. "
            f"model: {{name: yolo, model_name: yolov8n.yaml}}."
        )
    if objective.needs_num_classes:
        profile.require_num_classes(task.name)
    facts = profile.facts(task.name)
    chosen_stream = stream if stream is not None else topology.stream(task.input_topology)
    in_features = backbone.feature_dim(chosen_stream)
    out_features = objective.out_features(facts)
    head: Head
    if head_factory is not None:
        head = head_factory(in_features, _projected(task, out_features))
    elif prefer_native_head:
        native = backbone.native_head(chosen_stream, in_features, _projected(task, out_features))
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


def _projected(task: Task, out_features: int | None) -> int:
    """The width a *declared* head is built at, refused where the task projects nothing.

    ``out_features is None`` is metric learning's contract — the embedding is already the
    output — and only the topology's own head knows to answer that with an identity. A
    head named in config does not: as a zero it reached ``CosineHead(in_features, 0)`` and
    built a classifier with no prototypes at all, which fails several frames later and
    nowhere near the declaration that caused it.
    """
    if out_features is None:
        raise ValueError(
            f"Task '{task.name}' is supervised by comparison, so its embedding is the output and a "
            f"declared head has nothing to project onto. Drop 'head' / 'native_head' from the task."
        )
    return out_features
