"""A minimal, type-safe registry: pluggable components looked up by key.

## What belongs in a registry, and what does not

``Registry[T]`` says what the entries *are*. It cannot say which of them are there, and
for several ports that is a smaller set than "every implementation" — so the rule is
written once, here.

**A registry holds what a declaration *names*.** Everything such a class needs comes from
that declaration: its own values, the derived facts ``named_by`` offers, and any nested
component the declaration fills with ``_target_``. All three are registered —
``TimmBackbone(model_name, pretrained)`` on values alone,
``ExpectationCriterion(class_values, distance)`` on a derived fact plus a nested slot the
user writes as ``distance: {_target_: ...}``.

**What a declaration only *implies* is not registered**, because there is no name to
register it under — it is built from the declaration's shape, or from another section
existing at all. Two forms recur:

- a **composer**, built from a shape. ``WeightedSumCriterion(parts)`` is what ``loss:``
  *being a list* becomes; ``CompositeModel(backbone, components)`` is what ``model:``
  naming a **backbone** becomes.
- a **decorator**, built from another section. ``DistilledModel(student, teachers,
  criterion)`` exists because ``distillation:`` is present, not because anyone named it.

The line is not "does the constructor take a built object" — ``ExpectationCriterion``
takes a whole ``Criterion`` and is registered, because *config* built it. The line is who
does the building: a name in the declaration, or the assembler reading its shape.

Neither a composer nor a decorator is an alternative a user chooses between, so neither
has a name to be chosen by. Registering one would give a name to a default
(``model: {name: composite}``) or a second way to say what a section already says
(``model: {name: distilled}`` beside ``distillation:``).

Two things the rule does **not** say, because both look like counterexamples:

- Being constructed by assembly does not un-register anything. A *default* builds namable
  components directly — ``ContinuousObjective.build_criterion`` returns a
  ``WeightedSumCriterion`` over a ``CrossEntropyCriterion`` and an ``ExpectationCriterion``,
  and the latter two are registered because a user may also name them. Only the composer
  around them is nameless.
- Abstract bases are outside it entirely: ``WrappedCriterion`` is not registered because it
  is not a component, only the shape its subclasses share.

A registry is a convenience rather than a gate either way: anything unregistered is still
reachable by ``_target_``, which is exactly how a composer *would* be reached if someone
had one of their own.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator, Mapping
from inspect import signature
from typing import Any


class Registry[T]:
    """Maps hashable keys to component factories.

    The framework's single extension mechanism: each capability package declares its
    registries in ``<package>/registry.py``, implementations register at import time,
    and assembly resolves config names through them.

    A key is anything hashable — a short string for config-facing components, an enum
    member for axis behaviours, a tuple for composite dispatch.

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

    def register_instance(self, key: Hashable, instance: T) -> None:
        """Register a prebuilt object; ``create(key)`` returns it as-is.

        For components configured once and shared — axis behaviours, presets.
        """
        self._add(key, lambda: instance)

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
