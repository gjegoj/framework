"""One console for everything a run prints — including what the libraries it is built on print."""

from __future__ import annotations

from rich import get_console

from src.callbacks.dataset_summary import table_for
from src.callbacks.progress import MetricHistory, table
from src.console import HEADER_STYLE, console
from src.core import ClassDistribution


def test_the_framework_prints_through_the_console_every_rich_user_prints_through() -> None:
    """Two consoles on one terminal fight over the cursor: a live bar and a panel would garble each other.

    Lightning's own progress bar reaches for rich's console, so that one is what a run shares — and
    then a table drawn under the bar and a panel printed above it are the same display's business.
    """
    assert console() is get_console()


def test_every_table_a_run_prints_wears_the_same_header() -> None:
    """A run prints three tables one under another, and a header of its own in each reads as three tools.

    The model summary is Lightning's and already wears this; the two written here are what had to join
    it. Read off the tables rather than off a screen, because the style is what rich is handed and a
    terminal that renders no colour at all would hide a call site that forgot it.
    """
    drawn = [table(MetricHistory()), table_for("species", {"train": ClassDistribution({"cat": 1})}, {"train": 1})]

    assert {one.header_style for one in drawn} == {HEADER_STYLE}
