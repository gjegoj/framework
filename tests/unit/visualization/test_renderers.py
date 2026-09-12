"""One thing on a cell becomes markup: which colour, which corner, and what it says."""

from __future__ import annotations

import pytest

from src.visualization.entities import (
    Classification,
    Classifications,
    Label,
    Regression,
    Segmentation,
    SegmentationClass,
)
from src.visualization.palette import ink, task_palette
from src.visualization.renderers import (
    FieldContext,
    field_key,
    field_prefix,
    leaves_of,
    number,
    render_label,
    score_key,
    source_pill,
)
from tests.unit.visualization.test_overlays import block

PALETTE = task_palette("species", ("cat", "dog"))

EVERY_LABEL: tuple[Label, ...] = (
    Classification("cat", confidence=0.8),
    Classifications((Classification("cat", 0.9), Classification("dog", 0.2))),
    Regression(1.25),
    Segmentation((SegmentationClass("cat", block(8, 2, 2, 3)),)),
)
"""One of every arm of ``Label``. mypy holds the list complete; these hold each arm drawable."""


def context(side: str = "gt") -> FieldContext:
    return FieldContext(task="species", side=side, colors=PALETTE)  # type: ignore[arg-type]


class TestEveryLabelDraws:
    @pytest.mark.parametrize("label", EVERY_LABEL, ids=lambda one: type(one).__name__)
    def test_a_label_becomes_at_least_one_overlay_and_names_its_own_leaves(self, label: Label) -> None:
        items = render_label(label, context())

        assert items
        assert tuple(item.leaf for item in items) == leaves_of(label)
        assert all(item.key == field_key("species", "gt", item.leaf) for item in items)


class TestChips:
    def test_a_class_is_written_with_the_confidence_behind_it(self) -> None:
        (chip,) = render_label(Classification("cat", confidence=0.83), context())

        assert "cat 0.83" in chip.overlay

    def test_a_class_that_expressed_no_confidence_is_written_alone(self) -> None:
        (chip,) = render_label(Classification("cat"), context())

        assert 'data-full="cat"' in chip.overlay

    def test_truth_is_filled_and_a_prediction_is_outlined(self) -> None:
        """The two sides of one class share its hue; what tells them apart is which one is solid."""
        (truth,) = render_label(Classification("cat"), context("gt"))
        (predicted,) = render_label(Classification("cat"), context("pred"))

        assert f"background:{ink(PALETTE['cat'])}" in truth.overlay
        assert "background:" not in predicted.overlay
        assert "border-color:" in predicted.overlay

    def test_a_hesitant_prediction_is_edged_more_faintly_than_a_confident_one(self) -> None:
        (unsure,) = render_label(Classification("cat", confidence=0.1), context("pred"))
        (sure,) = render_label(Classification("cat", confidence=0.99), context("pred"))

        assert _rim_alpha(unsure.overlay) < _rim_alpha(sure.overlay)

    def test_a_number_is_shown_as_itself_in_the_one_colour_outside_every_palette(self) -> None:
        (chip,) = render_label(Regression(1.25), context())

        assert 'data-full="1.25"' in chip.overlay
        assert chip.leaf not in PALETTE

    def test_a_class_the_palette_does_not_know_still_draws(self) -> None:
        """A page shows what the model said; a class outside the vocabulary is a finding, not a crash."""
        (chip,) = render_label(Classification("bird"), context())

        assert chip.overlay

    def test_a_long_name_is_cut_on_the_chip_and_kept_whole_for_the_lightbox(self) -> None:
        name = "a" * 40
        (chip,) = render_label(Classification(name), FieldContext("species", "gt", PALETTE, max_chip_chars=8))

        assert f'data-full="{name}"' in chip.overlay
        assert f">{'a' * 7}…<" in chip.overlay

    def test_a_class_name_holding_a_quote_cannot_break_out_of_its_attribute(self) -> None:
        (chip,) = render_label(Classification('c"><script>'), context())

        assert "<script>" not in chip.overlay
        assert "&quot;" in chip.overlay


class TestMasks:
    def test_a_class_mask_is_an_image_laid_over_the_picture(self) -> None:
        (item,) = render_label(Segmentation((SegmentationClass("cat", block(8, 2, 2, 3)),)), context())

        assert item.zone == "cover"
        assert 'src="data:image/png;base64,' in item.overlay
        assert item.color == PALETTE["cat"]

    def test_chips_and_masks_go_to_different_corners_of_a_cell(self) -> None:
        (chip,) = render_label(Classification("cat"), context())

        assert chip.zone == "chips"


class TestKeys:
    def test_a_leaf_key_starts_with_the_prefixes_the_sidebar_selects_by(self) -> None:
        """The sidebar's branches switch their leaves by prefix, so the prefix has to be the key's own."""
        key = field_key("species", "gt", "cat")

        assert key.startswith(field_prefix("species", "gt"))
        assert key.startswith(field_prefix("species"))

    def test_a_task_and_a_measurement_together_name_one_slider(self) -> None:
        assert score_key("species", "iou") != score_key("species", "mae")


class TestNumber:
    @pytest.mark.parametrize(("value", "shown"), [(1.0, "1"), (0.123456, "0.123"), (12.5, "12.5"), (-0.0001, "-0")])
    def test_a_number_is_rounded_the_one_way_the_page_rounds(self, value: float, shown: str) -> None:
        assert number(value) == shown


def _rim_alpha(overlay: str) -> float:
    return float(overlay.split("border-color:rgba(")[1].split(")", maxsplit=1)[0].split(",")[3])


class TestSource:
    def test_a_local_path_is_offered_to_the_clipboard(self) -> None:
        """A file on the machine that trained is not a link anyone can follow from a browser."""
        pill = source_pill("/data/pet/7.png")

        assert 'class="src copy"' in pill and 'data-copy="/data/pet/7.png"' in pill
        assert "7.png" in pill

    def test_a_url_is_offered_as_a_link(self) -> None:
        pill = source_pill("https://example.test/7.png")

        assert 'href="https://example.test/7.png"' in pill and 'rel="noopener"' in pill
        assert "copy" not in pill

    def test_a_sample_that_came_from_nowhere_shows_nothing(self) -> None:
        assert source_pill(None) == ""
