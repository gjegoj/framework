"""A minimal, type-safe registry for pluggable components (Registry / Factory Method).

Replaces the over-engineered factory from the old prototype: no instance cache,
no predicate checkers, no config injection — just name → constructor lookup with
decorator registration. This is the single extension point users hit to plug in
their own backbones, heads, losses, metrics, etc.
"""

from __future__ import annotations

from collections.abc import Callable, KeysView


class Registry[T]:
    """Maps string keys to constructors so components plug in by name.

    Parameters:
        name (str): Human-readable registry name, used in error messages.

    Examples:
        >>> backbones: Registry[Backbone] = Registry("backbone")
        >>> @backbones.register("timm")
        ... class TimmBackbone(...): ...
        >>> model = backbones.create("timm", model_name="resnet18")
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, key: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Return a decorator that registers a class/factory under ``key``.

        Parameters:
            key (str): Lookup key; must be unique within this registry.

        Raises:
            ValueError: If ``key`` is already registered.
        """

        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if key in self._factories:
                raise ValueError(f"{self._name}: key {key!r} is already registered.")
            self._factories[key] = factory
            return factory

        return decorator

    def get(self, key: str) -> Callable[..., T]:
        """Return the constructor registered under ``key``.

        Raises:
            KeyError: If ``key`` is not registered.
        """
        try:
            return self._factories[key]
        except KeyError as error:
            available = sorted(self._factories)
            raise KeyError(f"{self._name}: unknown key {key!r}. Available: {available}.") from error

    def create(self, key: str, *args: object, **kwargs: object) -> T:
        """Construct the component registered under ``key`` with the given arguments."""
        return self.get(key)(*args, **kwargs)

    def __contains__(self, key: str) -> bool:
        return key in self._factories

    def keys(self) -> KeysView[str]:
        """Return the registered keys."""
        return self._factories.keys()
