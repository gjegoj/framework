"""``named_by``: what a callee is offered, and on which terms."""

from __future__ import annotations

from typing import Any

from src.core.registry import named_by


def sized(number: int, label: str = "x") -> tuple[int, str]:
    return number, label


def forwards_anything(**kwargs: Any) -> dict[str, Any]:
    return kwargs


def test_only_the_values_a_callee_names_are_offered() -> None:
    assert named_by(sized, {"number": 3, "unrelated": 7}) == {"number": 3}


def test_a_callee_that_sinks_kwargs_is_handed_nothing() -> None:
    """Matching is by name only: an upstream library must not receive framework facts."""
    assert named_by(forwards_anything, {"number": 3}) == {}


def test_nothing_offered_asks_no_signature() -> None:
    assert named_by(forwards_anything, {}) == {}
