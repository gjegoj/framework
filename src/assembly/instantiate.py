"""The one place that turns a declaration into a call: constructors and their arguments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hydra.utils import get_object

from src.config import ComponentConfig
from src.core.registry import named_by

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from src.core import Registry


def resolve_target(component: ComponentConfig, registry: Registry[Any] | None = None) -> Callable[..., Any]:
    """Return the constructor a component names, without calling it.

    Two forms reach a constructor and both come through here: a registry ``name`` —
    short and discoverable — and a dotted ``_target_`` import path, the escape hatch for
    code we do not own. Resolving them in one place is what keeps them from drifting
    apart semantically. Path resolution is delegated to ``hydra.utils.get_object``, so a
    bad ``_target_`` surfaces as Hydra's own ``ImportError``, honest because ``_target_``
    is Hydra's semantics.

    Separate from ``instantiate`` because some components must stay uninstantiated: an
    optimizer needs the model's parameters, which do not exist while config is read.

    Raises:
        LookupError: For a ``name`` component with no registry, or an unknown key.
        ImportError: If a ``_target_`` path cannot be imported.
    """
    if component.target is not None:
        return get_object(component.target)  # type: ignore[no-any-return]
    if registry is None:
        raise LookupError(
            f"Component '{component.name}' is a registry name, but no registry was given; "
            f"use a '{ComponentConfig.TARGET_KEY}' import path instead."
        )
    return registry.get(str(component.name))


def instantiate(component: ComponentConfig, registry: Registry[Any] | None = None, /, **derived: Any) -> Any:
    """Build the component a declaration names.

    ``component`` and ``registry`` are positional-only, so a derived value may
    be named anything without colliding with them.

    Parameters:
        component (ComponentConfig): What to build, and with which arguments.
        registry (Registry | None): Needed for the ``name`` form.
        **derived (Any): Values computed during assembly (schema- or
            data-derived). They win over config params on conflict: they come
            from the single source of truth, and a silent contradiction is how
            misalignment bugs start. They are *offered*, not forced — a factory
            that does not name one simply does not receive it, which lets a
            caller offer the same fact to a whole family of components without
            knowing which of them happens to want it.
    """
    factory = resolve_target(component, registry)
    params = {name: _resolve_value(value, derived) for name, value in component.params.items()}
    return factory(**{**params, **named_by(factory, derived)})


def _resolve_value(value: Any, derived: Mapping[str, Any]) -> Any:
    """Build nested components; walk lists; pass everything else through.

    A nested mapping is a component only when it carries ``_target_``: the ``name`` form
    needs a registry to mean anything, and a nested position has no registry context.

    Derived values travel down the whole tree on the same terms as at the top — each
    factory receives only what it names — because a nested component is often where the
    fact is actually wanted.
    """
    if isinstance(value, dict) and ComponentConfig.TARGET_KEY in value:
        return instantiate(ComponentConfig.model_validate(value), None, **derived)
    if isinstance(value, list):
        return [_resolve_value(item, derived) for item in value]
    return value
