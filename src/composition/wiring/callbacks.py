"""Callback wiring: turn the callbacks config into Lightning callbacks.

Construction is unified through ``instantiate`` — one grammar (registry key /
``name`` / ``_target_``) with a single home for import resolution. Callbacks that
need runtime/config context to build are handled by a Strategy registry of
*callback builders* keyed by registry name. The dispatch loop therefore stays
closed for modification (OCP): a new context-aware callback registers a builder
with ``@callback_builders.register(...)`` instead of editing ``build_callbacks``.

A builder has the signature ``(spec: Mapping, context: WiringContext) -> Callback``.
"""

from __future__ import annotations

from typing import Any

import lightning as L

from src.composition.wiring.common import WiringContext
from src.composition.wiring.tasks import resolve_num_classes
from src.config.schema import ExperimentConfig
from src.core.instantiate import instantiate, resolve_target
from src.core.registry import Registry
from src.core.runtime import RuntimeContext
from src.tasks.presets import task_presets
from src.transforms.batch.spec import TargetSpec

# key → builder ``(spec, context) -> Callback``; the registry's value type is the Callback it yields.
callback_builders: Registry[L.Callback] = Registry("callback_builder")


def build_callbacks(config: ExperimentConfig, runtime: RuntimeContext) -> list[Any]:
    """Build the ordered callback list from config.

    Each entry is normalised to an ``instantiate`` spec. A registered context-aware
    builder handles it when one exists for its key (e.g. ``checkpoint``,
    ``batch_transform``); otherwise it is built directly via ``instantiate`` against
    ``callback_registry``. YAML declaration order controls registration order — put
    ``ema`` before ``checkpoint`` so EMA weights are active when the checkpoint fires.

    Parameters:
        config (ExperimentConfig): Validated experiment config.
        runtime (RuntimeContext): Populated context (for builders needing num_classes).

    Returns:
        list: Lightning callbacks, ready for ``Trainer(callbacks=...)``.
    """
    if config.callbacks is None:
        return []

    context = WiringContext(config=config, runtime=runtime)
    callbacks: list[Any] = []
    for key, raw_params in config.callbacks.items():
        spec, registry_key = _normalize_callback_spec(key, raw_params)
        if registry_key is not None and registry_key in callback_builders:
            callbacks.append(callback_builders.create(registry_key, spec, context))
        else:
            callbacks.append(instantiate(spec, _callback_registry()))
    return callbacks


def _normalize_callback_spec(key: str, raw_params: dict[str, Any] | None) -> tuple[dict[str, Any], str | None]:
    """Resolve one callbacks entry into an ``(instantiate-spec, dispatch-key)`` pair.

    The dispatch key is the registry name used for builder lookup; it is ``None``
    for a ``_target_`` spec (which has no registry identity). The YAML block key is
    the default registry name when no explicit ``name`` is given.
    """
    params = dict(raw_params or {})
    if "_target_" in params:
        return params, None
    registry_key = str(params.pop("name", key))
    return {"name": registry_key, **params}, registry_key


def _callback_registry() -> Registry[L.Callback]:
    """Lazily fetch the callback registry (populated by importing the package)."""
    from src.callbacks.registry import callback_registry

    return callback_registry


# --------------------------------------------------------- context-aware builders


@callback_builders.register("checkpoint")
def _build_checkpoint(spec: dict[str, Any], context: WiringContext) -> L.Callback:
    """Default ``ModelCheckpoint.dirpath`` to ``{save_dir}/checkpoints`` when unset."""
    spec = dict(spec)
    if "dirpath" not in spec and context.config.save_dir is not None:
        spec["dirpath"] = f"{context.config.save_dir}/checkpoints"
    return instantiate(spec, _callback_registry())


@callback_builders.register("batch_transform")
def _build_batch_transform_callback(spec: dict[str, Any], context: WiringContext) -> L.Callback:
    """Build the nested ``transform`` (with runtime num_classes) before the callback."""
    spec = dict(spec)
    spec["transform"] = _build_batch_transform(spec["transform"], context)
    return instantiate(spec, _callback_registry())


def _build_batch_transform(spec: Any, context: WiringContext) -> Any:
    """Build a ``BatchTransform``, injecting every task's ``TargetSpec``.

    A batch transform changes the shared image, so it must rewrite *every* task's
    target. We inject the ``TargetSpec`` for all tasks; the topology/objective
    compatibility checks live in ``_instantiate_guarded`` (split out so they can be
    exercised directly against a hand-built ``TargetSpec`` list in tests).
    """
    targets = [_target_spec(task_name, context) for task_name in context.config.tasks]
    return _instantiate_guarded(spec, targets)


def _instantiate_guarded(spec: Any, targets: list[TargetSpec]) -> Any:
    """Resolve, guard, and instantiate a batch transform against explicit ``TargetSpec``s.

    Rejects (fails fast) a transform whose ``supported_topologies`` cannot cover some
    task's topology (e.g. MixUp — GLOBAL only — with a segmentation head), and one whose
    ``supported_objectives`` excludes some task's objective (e.g. label-mixing transforms
    with a METRIC/metric-learning task — mixed soft labels break proxy/margin losses).
    """
    from src.transforms.batch import batch_transforms

    transform_cls = _resolve_transform_class(spec, batch_transforms)
    name = getattr(transform_cls, "__name__", spec)

    supported_topologies: frozenset[Any] = getattr(transform_cls, "supported_topologies", frozenset())
    for target in targets:
        if target.topology not in supported_topologies:
            raise ValueError(
                f"batch transform {name!r} changes the shared image but cannot produce a coherent "
                f"target for task '{target.key}' ({target.topology}). It supports "
                f"{sorted(supported_topologies)}. Use a compatible transform or remove the incompatible head."
            )

    supported_objectives: frozenset[Any] | None = getattr(transform_cls, "supported_objectives", None)
    if supported_objectives is not None:
        excluded = [target for target in targets if target.objective not in supported_objectives]
        if excluded:
            excluded_keys = [target.key for target in excluded]
            offending_objectives = sorted({target.objective for target in excluded})
            raise ValueError(
                f"batch transform {name!r} changes the shared target but cannot support task(s) "
                f"{excluded_keys} (objective(s) {offending_objectives}). It supports objectives "
                f"{sorted(supported_objectives)}. Use a compatible transform or remove the incompatible head."
            )

    if isinstance(spec, str):
        spec = {"name": spec}
    return instantiate(spec, batch_transforms, targets=targets)


def _resolve_transform_class(spec: Any, registry: Registry[Any]) -> Any:
    """Resolve the transform's class from any spec form (string / name / ``_target_``)."""
    if isinstance(spec, str):
        return registry.get(spec)
    if "_target_" in spec:
        return resolve_target(str(spec["_target_"]))
    return registry.get(str(spec["name"]))


def _target_spec(task_name: str, context: WiringContext) -> TargetSpec:
    """Build the ``TargetSpec`` (key, topology, num_classes, objective) for one task."""
    task_config = context.config.tasks[task_name]
    preset = task_presets.create(task_config.preset)
    num_classes = resolve_num_classes(task_name, task_config, context.runtime)
    objective = preset.resolve_objective(task_config.objective)
    return TargetSpec(key=task_name, topology=preset.topology, num_classes=num_classes, objective=objective)
