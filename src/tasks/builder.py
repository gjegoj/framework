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
    from src.core.taxonomy import Objective, OutputTopology
    from src.tasks.topologies import TaskTopology


def build_task_components(
    task: Task,
    profile: DataProfile,
    backbone: Backbone,
    streams: tuple[str, ...] | None = None,
    prefer_native_head: bool = False,
    head_factory: Callable[[int | tuple[int, ...], int], Head] | None = None,
) -> TaskComponents:
    """Assemble the components that serve ``task`` inside a composite model.

    Resolves the task's axes to their behaviours, validates the pairing, reads inferred
    facts from ``profile`` when the objective needs them, and sizes the head from the
    backbone's stream — the ordering that keeps output sizes out of config.

    Parameters:
        task (Task): The declaration being served: its axes, name and weight.
        profile (DataProfile): The facts ``DataModule.setup`` recorded, read for this task.
        backbone (Backbone): Whose streams the head is sized from.
        streams (tuple[str, ...] | None): Read these streams instead of the topology's default
            or the backbone's pyramid.
        prefer_native_head (bool): Use the backbone's own head for the streams instead of a
            framework head; the default where the framework composes none.
        head_factory (Callable | None): Build this head instead of the topology's default,
            given ``(in_features, out_features)`` — a factory, because the sizes are resolved here.

    Raises:
        LookupError: If the native head is wanted but the backbone offers none for the streams,
            or a task reads a pyramid the backbone does not declare.
    """
    objective = objective_registry.create(task.objective)
    topology = topology_registry.create(task.output_topology)
    # Both halves of "can this framework serve this task?", asked together and before
    # anything is built.
    if not topology.supports(task.objective, task.input_topology):
        raise ValueError(
            f"Output topology '{task.output_topology}' with input topology '{task.input_topology}' "
            f"cannot be supervised by objective '{task.objective}'."
        )
    if objective.needs_num_classes:
        profile.require_num_classes(task.name)
    facts = profile.facts(task.name)
    streams = _streams_of(task, topology, backbone, streams)
    widths = tuple(backbone.feature_dim(name) for name in streams)
    in_features: int | tuple[int, ...] = widths[0] if len(widths) == 1 else widths
    out_features = objective.out_features(facts)
    head: Head
    if head_factory is not None:
        head = head_factory(in_features, _projected(task, out_features))
    elif prefer_native_head or not topology.composes_head:
        native = backbone.native_head(streams, in_features, _projected(task, out_features))
        if native is None:
            raise LookupError(f"{type(backbone).__name__} offers no native head for {', '.join(streams)}.")
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
        streams=streams,
        weight=task.weight,
    )


def _streams_of(
    task: Task, topology: TaskTopology, backbone: Backbone, declared: tuple[str, ...] | None
) -> tuple[str, ...]:
    """The streams this task's head reads: the config's, else the topology's, else the backbone's pyramid.

    Declared wins over derived, as everywhere. How many streams a head can read is the
    head's own business — a single-stream head handed several refuses itself by name.
    """
    streams = declared or topology.streams(task.input_topology) or backbone.pyramid()
    if not streams:
        raise LookupError(
            f"{type(backbone).__name__} declares no pyramid, and task '{task.name}' reads one: a "
            f"'{task.output_topology}' task needs a backbone with a detection head of its own."
        )
    return streams


def default_target_encoder(output_topology: OutputTopology, objective: Objective) -> str | None:
    """The encoder a task's target starts from when config declares none.

    The shape outranks the semantics: a dense cell is a mask file and an instances cell a
    list of objects whatever the labels mean; only a global cell asks the objective.
    ``None`` when neither axis has one — the caller owns the refusal. Here rather than in
    assembly because it is the same composition of the same two axes as ``build_task_components``.
    """
    shape = topology_registry.create(output_topology).default_target_encoder
    return shape or objective_registry.create(objective).default_target_encoder


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
