"""One console for everything the framework prints; the composition root and callbacks share it."""

from __future__ import annotations

from functools import cache

from rich.console import Console


@cache
def console() -> Console:
    return Console()
