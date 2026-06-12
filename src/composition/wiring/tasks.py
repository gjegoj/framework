"""Task-layer wiring: data codecs, target bindings, and task bundles."""

from __future__ import annotations

import dataclasses
from typing import Any

from src.config.schema import ExperimentConfig, TaskConfig
from src.core.entities import Task
from src.core.instantiate import instantiate
from src.core.keys import IMAGE
from src.core.runtime import RuntimeContext
from src.data.bindings import TargetBinding
from src.data.codecs import TargetCodec, target_codecs
from src.tasks.presets import task_presets
from src.tasks.strategies.objective import objective_strategies
from src.tasks.taxonomy import Topology


def _resolve_codec(task_cfg: TaskConfig) -> TargetCodec:
    """Build the data-layer target codec for one task.

    Priority: explicit ``target_codec:`` spec > the objective's ``default_codec``.
    The objective is the authority on label encoding, so the default follows from
    the resolved objective (preset default unless overridden in config).

    Parameters:
        task_cfg (TaskConfig): Validated task config.

    Returns:
        TargetCodec: An un-fitted codec ready for ``DataModule.setup``.
    """
    if task_cfg.target_codec is not None:
        return instantiate(task_cfg.target_codec, target_codecs)

    preset = task_presets.create(task_cfg.preset)
    if preset.default_codec is not None:
        codec_key = preset.default_codec
    else:
        objective = preset.resolve_objective(task_cfg.objective)
        codec_key = objective_strategies.create(objective).default_codec

    injected: dict[str, Any] = {}
    if task_cfg.class_mapping is not None:
        injected["class_mapping"] = task_cfg.class_mapping
    return instantiate(codec_key, target_codecs, **injected)


def build_bindings(config: ExperimentConfig) -> list[TargetBinding]:
    """Build target bindings (task name → column → codec) for all tasks.

    Called before ``DataModule.setup()`` — codecs are un-fitted here and fitted
    inside ``setup()``. The data-codec follows from the task's objective unless
    overridden by ``target_codec:`` in the task config.

    Parameters:
        config (ExperimentConfig): Validated experiment config.

    Returns:
        list[TargetBinding]: One binding per task, in declaration order.
    """
    return [
        TargetBinding(name=task_name, column=task_cfg.target, codec=_resolve_codec(task_cfg))
        for task_name, task_cfg in config.tasks.items()
    ]


def _resolve_num_classes(task_name: str, task_cfg: TaskConfig, runtime: RuntimeContext) -> int:
    """Return the concrete class count / output dim for a task.

    For regression tasks with ``dim`` set, returns ``dim`` directly.
    For all others tries ``num_classes`` from config then ``RuntimeContext``.
    """
    if task_cfg.dim is not None:
        return task_cfg.dim
    value = task_cfg.num_classes or runtime.num_classes.get(task_name)
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
    for task_name, task_cfg in config.tasks.items():
        num_classes = _resolve_num_classes(task_name, task_cfg, runtime)
        preset = task_presets.create(task_cfg.preset)
        task = preset.build(
            name=task_name,
            num_classes=num_classes,
            objective=task_cfg.objective,
            weight=task_cfg.weight,
            loss=task_cfg.loss,
            metrics=task_cfg.metrics,
            head=task_cfg.head,
            feature_key=task_cfg.feature_key,
        )
        task = _bind_input_keys(task, preset.topology, config.data.inputs)
        if task_cfg.class_mapping is not None:
            class_names = [task_cfg.class_mapping[i] for i in sorted(task_cfg.class_mapping)]
            task = dataclasses.replace(task, class_names=class_names)
        tasks.append(task)
    return tasks
