"""Where in a declaration a refusal came from, said one way so every position says it the same."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

KINDS = (LookupError, TypeError, ValueError)
"""What a refusal about a declaration is: a name nothing holds, a call that cannot be made, a value that
cannot stand. Ordered so that the narrowest a refusal is an instance of is the one it is restated as."""


@contextmanager
def naming(position: str) -> Iterator[None]:
    """Every refusal raised inside says which part of the declaration it is about.

    A registry lists every name it holds and a constructor lists every knob it takes, so what a reader
    is missing is rarely the alternatives — it is which line to change. A run writes the same handful of
    positions once per task, and the alternatives are identical for all of them, so the refusal alone
    cannot be acted on. Added by whoever knows the position rather than by whoever raises it: a builder
    is handed one declaration and cannot see where in a run it sat.

    The kind is kept, so a caller separating a name nothing implements from a declaration that cannot
    hold still separates them, and the cause is chained, so nothing of the original is lost. Where a
    declaration is what is being named, the position is spelled the way a run would override that line,
    which makes the message the path to it; where the data is, it is the split and the column the row
    came from.

    The kind rather than the class, because a class is not always something this can build: measured on
    pydantic 2.13.4, ``ValidationError`` is a ``ValueError`` whose constructor takes line errors rather
    than a message, and rebuilding one as itself answered with ``ValidationError.__new__() missing 1
    required positional argument`` — a refusal naming the declaration and the fix, replaced by the
    internals of whoever raised it. Every refusal this framework writes is one of the three below, and
    a library's own is caught by whichever of them it is.
    """
    try:
        yield
    except KINDS as error:
        kind = next(one for one in KINDS if isinstance(error, one))
        raise kind(f"{position}: {error}") from error
