"""``fill_signature``: the one exception to explicit facts, for constructors we do not own."""

from __future__ import annotations

from typing import Any

from src.config.instantiate import fill_signature


def sized(number: int, label: str = "x") -> tuple[int, str]:
    return number, label


def forwards_anything(**kwargs: Any) -> dict[str, Any]:
    return kwargs


def test_only_the_facts_a_constructor_names_are_filled() -> None:
    assert fill_signature(sized, number=3, unrelated=7) == {"number": 3}


def test_a_constructor_that_sinks_kwargs_is_handed_nothing() -> None:
    """Matching is by name only: an upstream library must not receive framework facts."""
    assert fill_signature(forwards_anything, number=3) == {}


def test_nothing_offered_asks_no_signature() -> None:
    assert fill_signature(forwards_anything) == {}
