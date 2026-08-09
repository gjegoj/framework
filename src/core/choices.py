"""Refusing a value that is not one of the few its parameter accepts."""

from __future__ import annotations

from typing import TYPE_CHECKING, get_args

if TYPE_CHECKING:
    from typing import TypeAliasType


def one_of[T: str](value: T, choices: TypeAliasType) -> T:
    """Return ``value`` when the ``choices`` alias admits it, and refuse it by name otherwise.

    ``Literal`` states the closed set and mypy enforces it, but it is erased at
    runtime — and a component's knobs arrive from YAML, where a typo is one keystroke
    away. Unchecked, a misspelt choice falls through to whichever branch the code
    writes last, so a run silently pools differently or reduces differently. (Config
    *sections* need none of this: they are pydantic and validate themselves.)

    The alias is the whole declaration — it carries the options *and* its own name, so
    a call site repeats neither::

        type Pooling = Literal["cls", "mean"]
        self._pooling = one_of(pooling, Pooling)

    which refuses ``"men"`` with ``Pooling must be one of cls, mean, got 'men'.``

    Name the alias after the parameter it types, so the message points at the key a
    user actually wrote in YAML — ``Reduction`` for ``reduction``.

    Parameters:
        value (T): What the caller was given.
        choices (TypeAliasType): The ``type X = Literal[...]`` alias declaring the set.

    Raises:
        ValueError: If ``value`` is not one of the alias's options.
    """
    options: tuple[str, ...] = get_args(choices.__value__)
    if value not in options:
        raise ValueError(f"{choices.__name__} must be one of {', '.join(options)}, got {value!r}.")
    return value
