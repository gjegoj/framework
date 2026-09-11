"""One console for everything the framework prints; whatever writes to a terminal shares it."""

from __future__ import annotations

from functools import cache

from rich.console import Console


@cache
def console() -> Console:
    return Console()
