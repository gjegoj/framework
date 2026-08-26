"""Refusing a value that is not one of the few its parameter accepts."""

from __future__ import annotations

from typing import TYPE_CHECKING, get_args

if TYPE_CHECKING:
    from typing import TypeAliasType


def one_of[T: str](value: T, choices: TypeAliasType) -> T:
    """Return ``value`` if the ``choices`` Literal alias admits it; refuse it by name otherwise.

    ``Literal`` is erased at runtime, and a knob arriving from YAML can be misspelt. Name the
    alias after the parameter (``Reduction`` for ``reduction``) so the message names the key::

        type Pooling = Literal["cls", "mean"]
        self._pooling = one_of(pooling, Pooling)
    """
    options: tuple[str, ...] = get_args(choices.__value__)
    if value not in options:
        raise ValueError(f"{choices.__name__} must be one of {', '.join(options)}, got {value!r}.")
    return value
