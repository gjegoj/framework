"""How this framework writes to a terminal: one console, one look."""

from __future__ import annotations

from functools import cache

from rich.console import Console

HEADER_STYLE = "bold magenta"
"""Column headers, inherited rather than chosen.

Lightning's ``RichModelSummary`` prints ``bold magenta`` headers and its table is the
first one a run shows, so matching it makes ours look deliberate rather than different.
"""

TITLE_STYLE = "bold cyan"
"""A table's own name. A different colour from the headers on purpose: in the same
one it reads as another header row rather than as the heading above them."""


@cache
def console() -> Console:
    """The console this framework prints through.

    One instance, because rich reads the terminal's width and capabilities when it is
    built, and two of them can disagree about the window they share. Three packages
    print through it — a callback's tables, the export verification's, and the progress
    bar's own — and three consoles with three styles read as three programs sharing a
    window.

    Cached rather than made a module-level global, so importing this costs nothing on a
    run that never prints.
    """
    return Console()
