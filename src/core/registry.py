"""A minimal, type-safe registry: pluggable components looked up by key.

The registration rule lives on ``Registry`` itself. Two things it does **not** say,
because both look like counterexamples:

- Being constructed by assembly does not un-register anything. A default may build
  registered components directly — ``ContinuousObjective.build_criterion`` returns a
  ``WeightedSumCriterion`` over two registered criteria; only the nameless composer
  around them stays out.
- Abstract bases are outside the rule entirely: ``WrappedCriterion`` is not a
  component, only the shape its subclasses share.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator, Mapping
from inspect import signature
from typing import Any, overload


class Registry[T]:
    """Maps hashable keys to component factories.

    The framework's single extension mechanism: each capability package declares its
    registries in ``<package>/registry.py``, implementations register at import time,
    and assembly resolves config names through them. A key is anything hashable — a
    short string for config-facing components, an enum member for axis behaviours, a
    tuple for composite dispatch.

    **A registry holds what a declaration *names*** — its own values, the derived
    facts ``named_by`` offers, a nested slot filled with ``_target_``. What a
    declaration only *implies* has no name to register under: a *composer* built
    from a section's shape (``WeightedSumCriterion`` is what ``loss:`` being a list
    becomes) and a *decorator* built from another section existing at all
    (``DistilledModel`` exists because ``distillation:`` is present). The line is
    who does the building — a name in the declaration, or the assembler reading its
    shape; registering the latter would name a default or say twice what a section
    already says. A registry is a convenience, not a gate: anything unregistered
    stays reachable by ``_target_``.

    Parameters:
        kind (str): What is being registered, as it should read in an error
            message — ``"criterion"``, ``"head"``.

    Examples:
        >>> criterion_registry: Registry[Criterion] = Registry("criterion")
        >>> @criterion_registry.register("cross_entropy")
        ... class CrossEntropyCriterion(WrappedCriterion): ...
        >>> criterion = criterion_registry.create("cross_entropy", label_smoothing=0.1)
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._factories: dict[Hashable, Callable[..., T]] = {}

    def register(self, key: Hashable) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Return a decorator that registers a class or factory under ``key``."""

        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            self._add(key, factory)
            return factory

        return decorator

    @overload
    def register_instance(self, key: Hashable, instance: T) -> None: ...

    @overload
    def register_instance(self, key: Hashable) -> Callable[[type[T]], type[T]]: ...

    def register_instance(self, key: Hashable, instance: T | None = None) -> Callable[[type[T]], type[T]] | None:
        """Register a prebuilt object; ``create(key)`` returns it as-is.

        For components configured once and shared — axis behaviours, presets. Two
        arities for one rule because Python decorates only a ``class`` or ``def``,
        never an expression: pass the instance when the value is built on the spot,
        or omit it to decorate a class whose no-argument construction *is* the
        instance. The decorator constructs eagerly, so a declaration that cannot
        build dies at import — the same moment a passed instance would have.
        """
        if instance is not None:
            self._add(key, lambda: instance)
            return None

        def decorator(cls: type[T]) -> type[T]:
            built = cls()
            self._add(key, lambda: built)
            return cls

        return decorator

    def create(self, key: Hashable, **kwargs: Any) -> T:
        """Build the component registered under ``key`` with ``kwargs``."""
        return self.get(key)(**kwargs)

    def get(self, key: Hashable) -> Callable[..., T]:
        """Return the factory registered under ``key``.

        Raises:
            LookupError: If ``key`` is unknown; names the kind and lists
                the registered keys.
        """
        try:
            return self._factories[key]
        except KeyError:
            known = ", ".join(sorted(str(existing) for existing in self._factories)) or "none"
            raise LookupError(f"Unknown {self._kind} '{key}'. Registered: {known}.") from None

    def __contains__(self, key: object) -> bool:
        return key in self._factories

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._factories)

    def _add(self, key: Hashable, factory: Callable[..., T]) -> None:
        if key in self._factories:
            raise ValueError(f"{self._kind} '{key}' is already registered.")
        self._factories[key] = factory


def named_by(callee: Callable[..., Any], offered: Mapping[str, Any]) -> dict[str, Any]:
    """The offered values ``callee`` names in its signature, and only those.

    How a component receives a fact assembly computed — ``num_classes``,
    ``class_values`` — without config having to restate it. Facts are *offered*: a
    factory that does not name one simply does not receive it, which lets a caller
    offer the same fact to a whole family of components without knowing which of
    them wants it.

    Matching is by name, never by ``**kwargs``, so a component that forwards
    unknown arguments to an upstream library is not handed framework facts it
    never asked for.

    Parameters:
        callee (Callable): Whose signature decides what it gets.
        offered (Mapping[str, Any]): Everything available to offer.
    """
    if not offered:
        return {}
    named = signature(callee).parameters
    return {name: value for name, value in offered.items() if name in named}
