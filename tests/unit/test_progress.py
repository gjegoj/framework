"""Progress reporting: visible to a person, invisible to a log file."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

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
