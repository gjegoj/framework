"""One constructor boundary for names and import paths; derived facts arrive from the caller, never from config."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from inspect import signature
from typing import Any

from src.config.schema import ComponentConfig
from src.core import Registry


def resolve_factory(component: ComponentConfig, registry: Registry[Any] | None = None) -> Callable[..., Any]:
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
    """Build a declared component with its arguments plus the facts the caller derived."""
    refuse_restated_facts(component, facts)
    return resolve_factory(component, registry)(**resolve_params(component), **facts)


def instantiate_offering(component: ComponentConfig, registry: Registry[Any] | None = None, /, **facts: Any) -> Any:
    """Build a declared component, *offering* derived facts: it receives the ones its signature names.

    The counterpart of ``instantiate``, which *imposes* them. A loss, a metric and a schedule are each
    built from a table of facts the run settled — and each takes only what it understands, so ``mae``
    stands beside ``accuracy`` without being handed a vocabulary it would refuse. A fact the declaration
    restates is still refused by name, so nothing is stated twice.
    """
    factory = resolve_factory(component, registry)
    offered = fill_signature(factory, **facts)
    refuse_restated_facts(component, offered)
    return factory(**resolve_params(component), **offered)


def refuse_restated_facts(component: ComponentConfig, facts: Iterable[str]) -> None:
    """Refuse a declaration that restates a fact the framework will hand this constructor itself.

    Sizes and vocabularies are stated once, where they are known — in the data, in the backbone, in the
    task — and never copied into a config, where the copy would be free to disagree with the original.
    """
    restated = sorted(component.params.keys() & set(facts))
    if restated:
        raise ValueError(
            f"{component.spelled!r} declares {', '.join(restated)}, which the framework derives; "
            "drop it from the declaration."
        )


def resolve_params(component: ComponentConfig) -> dict[str, Any]:
    """The declared arguments, with every nested ``_target_`` mapping already built."""
    return {name: _resolve_value(value) for name, value in component.params.items()}


def fill_signature(factory: Callable[..., Any], **facts: Any) -> dict[str, Any]:
    """Hand a constructor the framework does not own only the facts its signature names.

    The one bounded exception to explicit facts: a constructor the framework does not own — a
    torchmetrics metric, a torch schedule, a loss reached by ``_target_`` — takes what it understands
    and nothing else. One that forwards ``**kwargs`` names nothing and so receives nothing.
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
