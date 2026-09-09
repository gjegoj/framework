"""The one place that turns a declaration into a call: constructors and their arguments."""

from __future__ import annotations

from inspect import signature
from typing import TYPE_CHECKING, Any

from hydra.utils import get_object

from src.config.components import ComponentConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.core import Registry


def resolve_target(component: ComponentConfig, registry: Registry[Any] | None = None) -> Callable[..., Any]:
    """Return the constructor a component names, without calling it.

    A registry ``name`` and a dotted ``_target_`` import path both come through here, so
    the two cannot drift; path resolution is ``hydra.utils.get_object``, so a bad
    ``_target_`` surfaces as Hydra's own ``ImportError``. Separate from ``instantiate``
    because an optimizer needs the model's parameters, which do not exist yet.

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


def instantiate(component: ComponentConfig, registry: Registry[Any] | None = None) -> Any:
    """Build the component a declaration names, from that declaration alone.

    Nothing travels beside it: a derived fact (a head's sizes, a criterion's class count)
    is passed by the caller that knows it, through ``resolve_target`` and ``resolve_params``,
    and one written in config as well is refused by name (ADR-0004).

    Parameters:
        component (ComponentConfig): What to build, and with which arguments.
        registry (Registry | None): Needed for the ``name`` form.
    """
    return resolve_target(component, registry)(**resolve_params(component))


def resolve_params(component: ComponentConfig) -> dict[str, Any]:
    """The declared constructor arguments with every nested component built — for a caller that constructs itself."""
    return {name: _resolve_value(value) for name, value in component.params.items()}


def refuse_a_declared_fact(component: ComponentConfig, *facts: str) -> None:
    """A derived fact written in the declaration too — refused naming both.

    The one place the rule "declare once" is enforced for components (ADR-0004): the caller that
    passes a fact names it here, and a config carrying the same key dies while the
    experiment is built, naming the declaration and the fact.
    """
    written = [name for name in facts if name in component.params]
    if written:
        raise ValueError(
            f"'{component.spelled}' declares {', '.join(written)}, which the framework "
            f"derives (from the backbone and the data). Drop it from the declaration."
        )


def fill_signature(factory: Callable[..., Any], **facts: Any) -> dict[str, Any]:
    """The facts a constructor names in its signature, and only those.

    The one exception to explicit arguments, for constructors this framework does not own
    and cannot teach a facts object: a torchmetrics metric, a torch scheduler. Matched by
    name, never by ``**kwargs``, so a library forwarding unknown arguments upstream is not
    handed framework facts it never asked for. Our own components take their facts as
    ordinary arguments and never come through here.
    """
    if not facts:
        return {}
    named = signature(factory).parameters
    return {name: value for name, value in facts.items() if name in named}


def _resolve_value(value: Any) -> Any:
    """Build nested components; walk lists and mappings; pass everything else through.

    A nested mapping is a component only when it carries ``_target_``: the ``name`` form
    needs a registry to mean anything, and a nested position has no registry context. A
    nested component is built from its own declaration, like the top one; what it needs
    from the run reaches it through the component that holds it (a wrapper's
    ``with_geometry``), never by name from here. A mapping *of* components — a dual
    encoder's ``encoders: {image: {_target_: ...}, text: {_target_: ...}}`` — is walked
    like a list of them, keyed by name; a mapping of plain values comes back as it was.
    """
    if isinstance(value, dict) and ComponentConfig.TARGET_KEY in value:
        return instantiate(ComponentConfig.model_validate(value))
    if isinstance(value, dict):
        return {key: _resolve_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_value(item) for item in value]
    return value
