"""One console for everything a run prints — including what the libraries it is built on print."""

from __future__ import annotations

from rich import get_console

from src.console import console


def test_the_framework_prints_through_the_console_every_rich_user_prints_through() -> None:
    """Two consoles on one terminal fight over the cursor: a live bar and a panel would garble each other.

    Lightning's own progress bar reaches for rich's console, so that one is what a run shares — and
    then a table drawn under the bar and a panel printed above it are the same display's business.
    """
    assert console() is get_console()
