"""A minimal, type-safe registry: pluggable components looked up by key.

Being built by assembly does not un-register anything (a default may build registered
components directly); abstract bases such as ``WrappedCriterion`` are not components.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator, Mapping
from inspect import signature
from typing import Any, overload


class Registry[T]:
    """Maps hashable keys to component factories.

    The framework's single extension mechanism: packages declare registries in
    ``<package>/registry.py``, implementations register at import time, assembly resolves
    config names through them. A registry holds what a declaration *names*; what it only
    *implies* (a composer built from a section's shape) has no name to register under, and
    anything unregistered stays reachable by ``_target_``.

    Parameters:
        kind (str): What is being registered, as it reads in an error message — ``"criterion"``.
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

        For components configured once and shared — axis behaviours, presets. Pass the instance,
        or omit it to decorate a class whose no-argument construction *is* the instance; the
        decorator constructs eagerly, so a declaration that cannot build dies at import.
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

    How a component receives a fact assembly computed (``num_classes``) without config
    restating it: a factory that does not name a fact does not receive it. Matched by name,
    never by ``**kwargs``, so a component forwarding unknown arguments upstream is not handed
    framework facts it never asked for.
    """
    if not offered:
        return {}
    named = signature(callee).parameters
    return {name: value for name, value in offered.items() if name in named}
