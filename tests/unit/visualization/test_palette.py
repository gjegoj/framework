"""Class colours: the same class is the same colour everywhere, and a class's name can be read."""

from __future__ import annotations

import pytest

from src.visualization.palette import FALLBACK_COLOR, REGRESSION_COLOR, hex_to_rgb, ink, task_palette

READABLE = 4.5
"""WCAG's contrast ratio for text under 18px, which is what a chip is."""


def contrast(one: str, other: str) -> float:
    """The WCAG ratio, written here rather than imported: the requirement is the test's, not the code's.

    ``palette.py`` only picks a lightness. What that lightness has to achieve is stated in this file,
    in the formula the standard gives, so the implementation cannot satisfy the check by defining it.
    """
    lighter, darker = sorted((_luminance(one), _luminance(other)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def _luminance(color: str) -> float:
    def linear(channel: int) -> float:
        scaled = channel / 255
        return scaled / 12.92 if scaled <= 0.04045 else ((scaled + 0.055) / 1.055) ** 2.4

    red, green, blue = (linear(channel) for channel in hex_to_rgb(color))
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


WHEEL = (*task_palette("species", tuple(f"class{index}" for index in range(60))).values(),)
"""Sixty classes walk the hue circle at about six degrees a step — every hue a palette can produce."""


class TestTaskPalette:
    def test_every_class_gets_a_colour_of_its_own(self) -> None:
        classes = tuple(f"class{index}" for index in range(12))
        palette = task_palette("species", classes)

        assert set(palette) == set(classes)
        assert len(set(palette.values())) == len(classes)

    def test_a_class_keeps_its_colour_however_the_vocabulary_was_ordered(self) -> None:
        """A page is read beside another page of the same run; a reordered config must not recolour it."""
        assert task_palette("species", ("cat", "dog", "bird")) == task_palette("species", ("dog", "bird", "cat"))

    def test_two_tasks_do_not_paint_their_first_class_the_same(self) -> None:
        assert task_palette("species", ("a",))["a"] != task_palette("breed", ("a",))["a"]


class TestInk:
    @pytest.mark.parametrize("color", [*WHEEL, REGRESSION_COLOR, FALLBACK_COLOR])
    def test_any_colour_a_page_can_produce_is_readable_once_it_is_ink(self, color: str) -> None:
        """Both chips at once: contrast is symmetric, so one bound covers white-on-class and class-on-white.

        Over the whole circle rather than one page's classes — a run with a class count nobody has
        tried yet lands on hues nobody has looked at.
        """
        assert contrast(ink(color), "#ffffff") >= READABLE

    def test_ink_keeps_the_hue_it_was_given(self) -> None:
        """A chip and its swatch have to read as the same class; only the lightness may move."""
        red, green, blue = hex_to_rgb(ink("#39d0d0"))

        assert red < green and abs(green - blue) <= 1

    def test_the_palettes_own_colours_are_not_used_as_writing(self) -> None:
        """The reason two readings exist: the separating palette is unreadable at chip size."""
        assert min(contrast(color, "#ffffff") for color in WHEEL) < READABLE


class TestColours:
    @pytest.mark.parametrize(("value", "expected"), [("#000000", (0, 0, 0)), ("#ffffff", (255, 255, 255))])
    def test_a_hex_colour_reads_as_the_three_numbers_a_mask_is_painted_with(
        self, value: str, expected: tuple[int, int, int]
    ) -> None:
        assert hex_to_rgb(value) == expected
