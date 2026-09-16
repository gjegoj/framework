"""Where in a declaration a refusal came from, said one way so every position says it the same."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def naming(position: str) -> Iterator[None]:
    """Every refusal raised inside says which part of the declaration it is about.

    A registry lists every name it holds and a constructor lists every knob it takes, so what a reader
    is missing is rarely the alternatives — it is which line to change. A run writes the same handful of
    positions once per task, and the alternatives are identical for all of them, so the refusal alone
    cannot be acted on. Added by whoever knows the position rather than by whoever raises it: a builder
    is handed one declaration and cannot see where in a run it sat.

    The type is kept, so a caller catching ``LookupError`` still catches one, and the cause is chained,
    so nothing of the original is lost. Where a declaration is what is being named, the position is
    spelled the way a run would override that line, which makes the message the path to it; where the
    data is, it is the split and the column the row came from.
    """
    try:
        yield
    except (LookupError, ValueError, TypeError) as error:
        raise type(error)(f"{position}: {error}") from error
