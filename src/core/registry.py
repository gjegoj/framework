"""Names a declaration may write, mapped to the classes that serve them."""

from __future__ import annotations

from collections.abc import Callable, Iterator


class Registry[T]:
    """A package's catalogue: implementations register beside their definition, a config names one.

    A registry holds what a declaration *names*; what a declaration only *implies* (a composite
    built from a section's shape) has no name here, and anything unregistered stays reachable
    by ``_target_``. Construction is not this class's business: a resolved class is built by
    ``config.instantiate`` with the declaration's arguments, the same way a ``_target_`` is.

    Args:
        kind: What is registered, as it reads in an error message, e.g. ``"loss"``.
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, type[T]] = {}

    def register(self, name: str, *aliases: str) -> Callable[[type[T]], type[T]]:
        """Return a decorator registering a class under ``name`` and every alias."""

        def decorator(cls: type[T]) -> type[T]:
            for key in (name, *aliases):
                if key in self._entries:
                    raise ValueError(f"{self._kind} {key!r} is already registered.")
                self._entries[key] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        """Return the class registered under ``name``; an unknown name lists the known ones."""
        try:
            return self._entries[name]
        except KeyError:
            known = ", ".join(sorted(self._entries)) or "none"
            raise LookupError(f"Unknown {self._kind} {name!r}. Registered: {known}.") from None

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._entries))
