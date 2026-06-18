"""Task-layer wiring: data encoders, target bindings, and task bundles."""

from __future__ import annotations

import dataclasses
from typing import Any

from src.config.schema import ExperimentConfig, TaskConfig
from src.core.entities import Task
from src.core.instantiate import instantiate
from src.core.keys import IMAGE
from src.core.runtime import RuntimeContext
from src.data.bindings import TargetBinding
from src.data.encoders import TargetEncoder
from src.data.registry import target_encoders
from src.tasks.presets import task_presets
from src.tasks.strategies.objective import objective_strategies
from src.tasks.taxonomy import Topology


def _resolve_encoder(task_config: TaskConfig) -> TargetEncoder:
    """Build the data-layer target encoder for one task.

    Priority: explicit ``target_encoder:`` spec > the objective's ``default_encoder``.
    The objective is the authority on label encoding, so the default follows from
    the resolved objective (preset default unless overridden in config).

    Parameters:
        task_config (TaskConfig): Validated task config.

    Returns:
        TargetEncoder: An un-fitted encoder ready for ``DataModule.setup``.
    """
    if task_config.target_encoder is not None:
        return instantiate(task_config.target_encoder, target_encoders)

    preset = task_presets.create(task_config.preset)
    if preset.default_encoder is not None:
        encoder_key = preset.default_encoder
    else:
        objective = preset.resolve_objective(task_config.objective)
        encoder_key = objective_strategies.create(objective).default_encoder

    injected: dict[str, Any] = {}
    if task_config.class_mapping is not None:
        injected["class_mapping"] = task_config.class_mapping
    return instantiate(encoder_key, target_encoders, **injected)


def build_bindings(config: ExperimentConfig) -> list[TargetBinding]:
    """Build target bindings (task name → column → encoder) for all tasks.

    Called before ``DataModule.setup()`` — encoders are un-fitted here and fitted
    inside ``setup()``. The data-encoder follows from the task's objective unless
    overridden by ``target_encoder:`` in the task config.

    Parameters:
        config (ExperimentConfig): Validated experiment config.

    Returns:
        list[TargetBinding]: One binding per task, in declaration order.
    """
    return [
        TargetBinding(name=task_name, column=task_config.target, encoder=_resolve_encoder(task_config))
        for task_name, task_config in config.tasks.items()
    ]


def _resolve_num_classes(task_name: str, task_config: TaskConfig, runtime: RuntimeContext) -> int:
    """Return the concrete class count / output dim for a task.

    For regression tasks with ``dim`` set, returns ``dim`` directly.
    For all others tries ``num_classes`` from config then ``RuntimeContext``.
    """
    if task_config.dim is not None:
        return task_config.dim
    value = task_config.num_classes or runtime.num_classes.get(task_name)
    if value is None:
        raise ValueError(
            f"num_classes for task '{task_name}' is not set in config and could not be "
            "inferred from data. Ensure DataModule.setup() ran before build_tasks(), "
            "or set num_classes / dim explicitly in the task config."
        )
    return value


def _input_aliases(inputs: str | dict[str, Any]) -> tuple[str, ...]:
    """Extract ordered input alias names from the data config.

    For ``inputs: image_path`` (single-image shorthand) the only alias is the
    canonical ``IMAGE`` key.  For ``inputs: {anchor: ..., positive: ...}`` the
    aliases are the dict keys in declaration order.
    """
    if isinstance(inputs, dict):
        return tuple(inputs.keys())
    return (IMAGE,)


def _bind_input_keys(task: Task, topology: Topology, inputs: str | dict[str, Any]) -> Task:
    """Fill a multi-input task's key field (``view_keys``/``stream_keys``) from data.inputs.

    The data config is the single source of truth for input alias names, so
    RANKING (views through one shared backbone) and MULTISTREAM (streams from N
    separate encoders) presets leave the field ``None`` and it is derived here.
    Other topologies and already-populated specs pass through unchanged.
    """
    if topology == Topology.RANKING and task.head_spec.view_keys is None:
        head_spec = dataclasses.replace(task.head_spec, view_keys=_input_aliases(inputs))
    elif topology == Topology.MULTISTREAM and task.head_spec.stream_keys is None:
        head_spec = dataclasses.replace(task.head_spec, stream_keys=_input_aliases(inputs))
    else:
        return task
    return dataclasses.replace(task, head_spec=head_spec)


def build_tasks(config: ExperimentConfig, runtime: RuntimeContext) -> list[Task]:
    """Build task bundles after ``DataModule.setup()`` has populated ``RuntimeContext.num_classes``.

    For multi-input topologies (RANKING views / MULTISTREAM streams) whose key
    field is ``None`` (the default for config-driven experiments), the input
    alias names are derived here from ``config.data.inputs`` — the data config is
    the single source of truth.

    Parameters:
        config (ExperimentConfig): Validated experiment config.
        runtime (RuntimeContext): Populated context (num_classes must be set for each task).

    Returns:
        list[Task]: Assembled task bundles in config declaration order.

    Raises:
        ValueError: If num_classes for any task cannot be resolved.
    """
    tasks: list[Task] = []
    for task_name, task_config in config.tasks.items():
        num_classes = _resolve_num_classes(task_name, task_config, runtime)
        preset = task_presets.create(task_config.preset)
        task = preset.build(
            name=task_name,
            num_classes=num_classes,
            objective=task_config.objective,
            weight=task_config.weight,
            loss=task_config.loss,
            metrics=task_config.metrics,
            head=task_config.head,
            feature_key=task_config.feature_key,
        )
        task = _bind_input_keys(task, preset.topology, config.data.inputs)
        if task_config.class_mapping is not None:
            class_names = [task_config.class_mapping[i] for i in sorted(task_config.class_mapping)]
            task = dataclasses.replace(task, class_names=class_names)
        tasks.append(task)
    return tasks
