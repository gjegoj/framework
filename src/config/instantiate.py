"""One constructor boundary for names and import paths; derived facts arrive from the caller, never from config."""

from __future__ import annotations

from collections.abc import Callable
from inspect import signature
from typing import Any

from src.config.schema import ComponentConfig
from src.core import Registry


def resolve_target(component: ComponentConfig, registry: Registry[Any] | None = None) -> Callable[..., Any]:
    """Return the class or factory a declaration names, resolving only that one implementation."""
    if component.import_path is not None:
        return _locate(component.import_path)
    if registry is None:
        raise LookupError(
            f"{component.name!r} is a registry name, but this position has no registry; "
            f"use a {ComponentConfig.TARGET_KEY!r} import path instead."
        )
    return registry.get(str(component.name))


def instantiate(component: ComponentConfig, registry: Registry[Any] | None = None, /, **facts: Any) -> Any:
    """Build a declared component with its arguments plus the facts the caller derived.

    A fact the declaration restates is refused by name: sizes and vocabularies are stated
    once, where they are known, and never copied into a config.
    """
    restated = sorted(facts.keys() & component.params.keys())
    if restated:
        raise ValueError(
            f"{component.spelled!r} declares {', '.join(restated)}, which the framework derives; "
            "drop it from the declaration."
        )
    return resolve_target(component, registry)(**resolve_params(component), **facts)


def resolve_params(component: ComponentConfig) -> dict[str, Any]:
    """The declared arguments, with every nested ``_target_`` mapping already built."""
    return {name: _resolve_value(value) for name, value in component.params.items()}


def fill_signature(factory: Callable[..., Any], **facts: Any) -> dict[str, Any]:
    """Hand a constructor the framework does not own only the facts its signature names.

    The one bounded exception to explicit facts, for torchmetrics metrics and torch schedulers;
    a constructor that forwards ``**kwargs`` names nothing and receives nothing.
    """
    named = signature(factory).parameters
    return {name: value for name, value in facts.items() if name in named}


def _resolve_value(value: Any) -> Any:
    if isinstance(value, dict) and ComponentConfig.TARGET_KEY in value:
        return instantiate(ComponentConfig.model_validate(value))
    if isinstance(value, dict):
        return {key: _resolve_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_value(item) for item in value]
    return value


def _locate(path: str) -> Callable[..., Any]:
    from hydra.utils import get_object

    try:
        target = get_object(path)
    except (ImportError, AttributeError, ValueError) as error:
        raise LookupError(f"Cannot resolve {path!r}: {error}") from error
    if not callable(target):
        raise TypeError(f"{path!r} is not a constructor.")
    return target  # type: ignore[no-any-return]
