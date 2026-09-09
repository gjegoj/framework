"""Progress reporting: visible to a person, invisible to a log file."""

from __future__ import annotations

import io
import itertools
from collections.abc import Iterator

import pytest
from rich.console import Console

from src.progress import track


def test_every_item_reaches_the_caller() -> None:
    assert list(track(range(5), "counting", total=5)) == [0, 1, 2, 3, 4]


def test_nothing_is_printed_when_no_terminal_is_watching(capsys: pytest.CaptureFixture[str]) -> None:
    """Tests and CI capture stdout; a bar there is noise nobody reads."""
    list(track(range(3), "counting", total=3))

    assert capsys.readouterr().out == ""


def test_an_unknown_total_is_allowed() -> None:
    assert list(track(iter("abc"), "counting")) == ["a", "b", "c"]


def test_items_are_yielded_lazily() -> None:
    """A caller that stops early must not have driven the whole iterable."""
    seen: list[int] = []

    def counted() -> Iterator[int]:
        for value in range(100):
            seen.append(value)
            yield value

    next(iter(track(counted(), "counting", total=100)))

    assert len(seen) == 1


def test_a_watching_terminal_sees_the_bar_and_still_gets_every_item(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a terminal the description, the count and the live status are drawn; the items pass through unchanged."""
    drawn = io.StringIO()
    monkeypatch.setattr("src.progress.console", lambda: Console(force_terminal=True, file=drawn, width=60))
    asked = itertools.count()

    items = list(track(range(3), "counting", total=3, status=lambda: f"asked {next(asked)}"))

    assert items == [0, 1, 2]
    shown = drawn.getvalue()
    assert "counting" in shown
    assert "3/3" in shown
    assert "asked 3" in shown  # the status is asked again after every item, not once at the start
