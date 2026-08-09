"""Deterministic, maximally-distinct class colours per task."""

from __future__ import annotations

from src.visualization.palette import hex_to_rgb, task_palette


def test_the_same_input_always_yields_the_same_colours() -> None:
    """Colours are identity across runs: a class must look the same in every report."""
    assert task_palette("species", ["cat", "dog"]) == task_palette("species", ["cat", "dog"])


def test_declaration_order_does_not_change_a_class_colour() -> None:
    """The palette sorts internally, so config reordering cannot recolour a report."""
    assert task_palette("species", ["dog", "cat"]) == task_palette("species", ["cat", "dog"])


def test_classes_of_one_task_get_distinct_colours() -> None:
    colours = task_palette("species", [f"class_{index}" for index in range(12)])

    assert len(set(colours.values())) == 12


def test_two_tasks_start_from_different_hues() -> None:
    """Each task seeds its own offset, so two single-class tasks do not collide."""
    assert task_palette("species", ["a"])["a"] != task_palette("breed", ["a"])["a"]


def test_hex_round_trips_to_rgb() -> None:
    assert hex_to_rgb("#ff8000") == (255, 128, 0)
