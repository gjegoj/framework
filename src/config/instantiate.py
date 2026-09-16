"""One constructor boundary for names and import paths; derived facts arrive from the caller, never from config."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from inspect import Parameter, signature
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
    factory = resolve_factory(component, registry)
    refuse_what_the_constructor_does_not_name(component, factory, imposed=facts)
    return factory(**resolve_params(component), **facts)


def instantiate_offering(component: ComponentConfig, registry: Registry[Any] | None = None, /, **facts: Any) -> Any:
    """Build a declared component, *offering* derived facts: it receives the ones its signature names.

    The counterpart of ``instantiate``, which *imposes* them. A loss, a metric and a schedule are each
    built from a table of facts the run settled — and each takes only what it understands, so ``mae``
    stands beside ``accuracy`` without being handed a vocabulary it would refuse. A fact the declaration
    restates is still refused by name, so nothing is stated twice — against the whole table rather than
    the part this constructor happens to name, because one that forwards everything would otherwise take
    the written copy while the derived one was never offered, and the copy would win without a word.

    A constructor that wants *more* than the table holds is refused here rather than by each caller:
    this is the function that knows what was offered, so it is the one that can say what was missing.
    """
    factory = resolve_factory(component, registry)
    refuse_restated_facts(component, facts)
    refuse_what_the_constructor_does_not_name(component, factory)
    offered = fill_signature(factory, **facts)
    try:
        return factory(**resolve_params(component), **offered)
    except TypeError as error:
        settled = ", ".join(sorted(facts)) or "nothing"
        raise ValueError(
            f"{component.spelled!r} needs more than this task settles about its target ({settled}): {error}"
        ) from error


def refuse_what_the_constructor_does_not_name(
    component: ComponentConfig, factory: Callable[..., Any], *, imposed: Iterable[str] = ()
) -> None:
    """Refuse a declaration its own constructor cannot take, before it is called rather than by its error.

    Both sides meet here and nowhere else: what a run wrote, and what the framework is about to hand
    over itself. Python's own ``TypeError`` carries neither — it says a keyword was unexpected, with no
    declaration in it and no word about who passed it — and read after the fact it cannot tell a
    misspelled knob from a fact the run never settled, which is how ``gama`` came to be reported as
    something to go and declare.

    A constructor forwarding ``**kwargs`` is taken at its word and left alone. Measured on this tree's
    own dependencies: every torchmetrics metric, ``smp.create_model`` and an albumentations chain
    forward what they are handed and each refuses its own unknown knobs in its own words, so a rule
    here would refuse declarations those libraries accept. One that says nothing at all is left alone
    for the same reason.
    """
    named = what_it_names(factory)
    if named is None or any(one.kind is Parameter.VAR_KEYWORD for one in named.values()):
        return
    accepted = ", ".join(sorted(named)) or "nothing"
    if unknown := sorted(component.params.keys() - named.keys()):
        raise ValueError(f"{component.spelled!r} takes no {', '.join(unknown)}; it takes {accepted}.")
    if refused := sorted(set(imposed) - named.keys()):
        raise ValueError(
            f"The framework hands {component.spelled!r} {', '.join(f'{one}=' for one in refused)} and its "
            f"constructor does not name {'it' if len(refused) == 1 else 'them'}; it takes {accepted}."
        )


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
    named = what_it_names(factory)
    return {} if named is None else {name: value for name, value in facts.items() if name in named}


def what_it_names(factory: Callable[..., Any]) -> Mapping[str, Parameter] | None:
    """The parameters a constructor declares, or ``None`` where it declares nothing readable.

    One reading for the two questions asked of a signature here — what may be handed over, and what may
    not be declared — so they cannot part over a constructor neither can read. Measured: ``inspect``
    refuses ``dict``, ``max`` and ``zip`` with ``no signature found for builtin``, and a check that
    read one would refuse, in the words of the reading, a declaration the constructor accepts.
    """
    try:
        return signature(factory).parameters
    except ValueError:
        return None


def _resolve_value(value: Any) -> Any:
    if isinstance(value, dict) and ComponentConfig.TARGET_KEY in value:
        return instantiate(ComponentConfig.model_validate(value))
    if isinstance(value, dict):
        _refuse_a_registry_name_below_the_position_a_registry_serves(value)
        return {key: _resolve_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_value(item) for item in value]
    return value


def _refuse_a_registry_name_below_the_position_a_registry_serves(value: dict[str, Any]) -> None:
    """A nested mapping spelling ``name`` meant to be a component, where only an import path can be one.

    A registry belongs to a position — this section is a loss, that one a head — and a nested argument
    has none, so the same spelling one level down resolves to nothing and travels on as the two-key
    mapping it literally is. No constructor expects that and none says so, which made a declaration read
    as a component by whoever wrote it and as a dictionary by everything that read it.
    """
    if "name" in value:
        raise ValueError(
            f"A nested 'name' names nothing: a registry serves the position a section declares and not the "
            f"arguments below it. Write {ComponentConfig.TARGET_KEY!r} with an import path here, or move the "
            f"component up to a position of its own."
        )


def _locate(path: str) -> Callable[..., Any]:
    from hydra.utils import get_object

    try:
        target = get_object(path)
    except (ImportError, AttributeError, ValueError) as error:
        raise LookupError(f"Cannot resolve {path!r}: {error}") from error
    if not callable(target):
        raise TypeError(f"{path!r} is not a constructor.")
    return target  # type: ignore[no-any-return]
