"""Views into one page: what it shows, what it marks as a mistake, and what narrows it."""

from __future__ import annotations

import html as escaping
import json
import re

import numpy as np
import pytest

from src.visualization.entities import (
    Classification,
    Image,
    SampleView,
    Score,
    Segmentation,
    SegmentationClass,
    Verdict,
)
from src.visualization.html import HtmlRenderer
from src.visualization.palette import ink, task_palette
from src.visualization.renderers import score_key
from tests.support.pixels import decoded


def view(
    said: str = "cat", true: str = "cat", correct: bool | None = None, scores: tuple[Score, ...] = ()
) -> SampleView:
    image = np.full((4, 6, 3), 120, dtype=np.uint8)
    return SampleView(
        image=Image(pixels=image, source="/data/1.png"),
        fields={("species", "gt"): Classification(true), ("species", "pred"): Classification(said)},
        verdicts={"species": Verdict(correct=correct, scores=scores)},
    )


def page(*views: SampleView, classes: dict[str, tuple[str, ...]] | None = None) -> str:
    return HtmlRenderer().render(list(views), title="samples/val", classes=classes or {})


class TestThePage:
    def test_one_cell_per_view(self) -> None:
        assert page(view(), view(), view()).count('class="cell') == 3

    def test_it_carries_its_own_styling_and_behaviour_and_fetches_nothing(self) -> None:
        """It is read inside a tracker's panel, where a second request may not be allowed at all."""
        shown = page(view())

        assert "<style>" in shown and "<script>" in shown
        assert "<script src" not in shown and "<link" not in shown

    def test_a_page_with_nothing_on_it_is_still_a_page(self) -> None:
        assert "<html" in page()

    def test_the_title_is_shown_and_escaped(self) -> None:
        shown = HtmlRenderer().render([view()], title="a<b", classes={})

        assert "a&lt;b" in shown and "a<b" not in shown


class TestVerdicts:
    def test_a_sample_every_task_got_right_is_marked_correct(self) -> None:
        shown = page(view(correct=True))

        assert "cell ok" in shown
        assert "✓ correct" in shown

    def test_a_sample_any_task_got_wrong_is_marked_a_mistake(self) -> None:
        shown = page(view(said="dog", correct=False))

        assert "cell bad" in shown
        assert "✗" in shown

    def test_a_sample_nothing_judged_is_neither(self) -> None:
        """A regression is not right or wrong; marking it either way would be an invented answer."""
        shown = page(view(correct=None))

        assert "cell ok" not in shown and "cell bad" not in shown

    def test_the_filter_reads_the_same_verdicts_the_badge_shows(self) -> None:
        shown = page(view(correct=False))

        assert json.loads(_attribute(shown, "data-verdicts")) == {"species": "wrong"}

    def test_a_task_with_no_yes_or_no_answer_stays_out_of_the_filter(self) -> None:
        assert json.loads(_attribute(page(view(correct=None)), "data-verdicts")) == {}

    def test_a_sample_that_matched_is_filed_as_correct(self) -> None:
        """The page's script compares against this very word; a badge saying one thing and the filter
        another would empty the `correct` view while every cell still shows a tick."""
        assert json.loads(_attribute(page(view(correct=True)), "data-verdicts")) == {"species": "correct"}


class TestScores:
    def test_a_cell_carries_every_number_measured_on_it(self) -> None:
        shown = page(view(scores=(Score("iou", 0.5), Score("mae", 2.0))))

        assert json.loads(_attribute(shown, "data-scores")) == {
            score_key("species", "iou"): 0.5,
            score_key("species", "mae"): 2.0,
        }

    def test_a_cell_and_the_slider_over_it_round_the_same_way(self) -> None:
        """They did not, and the lowest-scoring sample then failed its own floor and vanished —
        the one sample the sliders exist to find."""
        shown = page(view(scores=(Score("iou", 0.123456789),)), view(scores=(Score("iou", 0.9),)))

        carried = json.loads(_attribute(shown, "data-scores"))[score_key("species", "iou")]
        floor = float(_first(r'class="edge low" value="([-\d.e]+)"', shown))

        assert carried == 0.123, "rounded on the way to the page, not carried at full precision"
        assert floor == carried

    def test_a_measurement_that_diverged_is_printed_but_not_carried(self) -> None:
        """``NaN`` is spelled bare in JSON, which the page's parser rejects — one diverged number
        would take down every filter at once, not only its own slider."""
        shown = page(view(scores=(Score("iou", float("nan")),)))

        assert json.loads(_attribute(shown, "data-scores")) == {}
        assert "iou nan" in shown

    def test_a_slider_covers_the_range_the_page_holds_rather_than_the_one_it_could(self) -> None:
        """An overlap column between 0.55 and 0.71 gets its resolution where the samples are."""
        shown = page(view(scores=(Score("iou", 0.55),)), view(scores=(Score("iou", 0.71),)))

        assert 'min="0.55"' in shown and 'max="0.71"' in shown

    def test_a_page_that_measured_nothing_has_no_sliders(self) -> None:
        assert 'class="filter range"' not in page(view(correct=True))


class TestColours:
    def test_a_class_keeps_its_colour_on_a_page_that_shows_fewer_classes(self) -> None:
        """A palette walks the hue circle in class order, so seeding it from the classes on *this*
        page would recolour everything the moment a prediction introduces or drops one."""
        shown = page(view(), classes={"species": ("bird", "cat", "dog")})

        assert ink(task_palette("species", ("bird", "cat", "dog"))["cat"]) in shown

    def test_a_task_that_declared_no_vocabulary_is_coloured_by_what_is_shown(self) -> None:
        shown = page(view())

        assert ink(task_palette("species", ("cat",))["cat"]) in shown


class TestSidebar:
    def test_a_branch_per_task_and_a_row_per_leaf(self) -> None:
        shown = page(view(said="dog"))

        assert shown.count('class="node task"') == 1
        assert shown.count('class="node kind"') == 2
        assert shown.count('class="row"') == 2

    def test_truth_and_prediction_are_shown_as_two_rows_of_chips(self) -> None:
        """One filled chip beside one outlined chip, at 11px, in one hue, is easy to misread."""
        shown = page(view())

        assert shown.count('<span class="kind">') == 2


class TestBounds:
    def test_a_bound_that_could_only_draw_nothing_is_refused_by_the_page_that_owns_it(self) -> None:
        with pytest.raises(ValueError, match="max_side"):
            HtmlRenderer(max_side=0)

    def test_inlining_whole_is_said_with_none_rather_than_with_zero(self) -> None:
        assert HtmlRenderer(max_side=None).render([view()], title="t", classes={})


def _attribute(shown: str, name: str) -> str:
    return escaping.unescape(_first(rf"{name}=\"([^\"]*)\"", shown))


def _first(pattern: str, shown: str) -> str:
    """The first capture, or a failure that says what was looked for rather than one about None."""
    found = re.search(pattern, shown)
    assert found is not None, f"the page carries nothing matching {pattern}"
    return found.group(1)


class TestBoundsReachThePixels:
    """The knobs are refused when impossible; these hold that they also do something."""

    @staticmethod
    def big(side: int = 64) -> SampleView:
        pixels = np.random.default_rng(0).integers(0, 256, size=(side, side, 3), dtype=np.uint8)
        mask = np.zeros((side, side), dtype=bool)
        mask[side // 4 : side // 2, side // 4 : side // 2] = True
        return SampleView(
            image=Image(pixels=pixels),
            fields={("mask", "gt"): Segmentation((SegmentationClass("pet", mask),))},
        )

    def test_a_image_reaches_the_page_no_larger_than_the_bound(self) -> None:
        """What the bound is for: a cell inlines its image and one layer per class per side, so the
        page weighs the product of the three."""
        shown = HtmlRenderer(max_side=16).render([self.big()], title="t", classes={})

        assert decoded(_first(r'class="image" alt="sample" src="([^"]+)"', shown)).shape == (16, 16, 3)

    def test_a_mask_reaches_it_at_the_same_size_as_the_image_it_explains(self) -> None:
        """Off by a row and the overlay stops landing on the pixels it is about."""
        shown = HtmlRenderer(max_side=16).render([self.big()], title="t", classes={})

        assert decoded(_first(r'class="layer mask" data-key="[^"]*" src="([^"]+)"', shown)).shape == (16, 16, 4)

    def test_inlining_whole_leaves_the_pixels_as_they_are(self) -> None:
        shown = HtmlRenderer(max_side=None).render([self.big()], title="t", classes={})

        assert decoded(_first(r'class="image" alt="sample" src="([^"]+)"', shown)).shape == (64, 64, 3)


class TestTheFrame:
    def test_a_portrait_image_keeps_its_shape_inside_a_square_cell(self) -> None:
        """``aspect-ratio`` alone does not survive the clamp: the image stretches into the square and
        every mask stretches with it, which makes nothing look wrong."""
        tall = SampleView(image=Image(pixels=np.zeros((8, 4, 3), dtype=np.uint8)))

        assert "aspect-ratio:0.5;width:50%" in HtmlRenderer().render([tall], title="t", classes={})

    def test_a_landscape_image_fills_the_width(self) -> None:
        wide = SampleView(image=Image(pixels=np.zeros((4, 8, 3), dtype=np.uint8)))

        assert "aspect-ratio:2;width:100%" in HtmlRenderer().render([wide], title="t", classes={})


class TestTheMarkup:
    def test_every_tag_the_page_opens_it_closes(self) -> None:
        """Nothing else here reads the page as markup: the assertions above are substrings, so a tag
        left open would reach a browser before it reached a test."""
        from html.parser import HTMLParser

        void = {"img", "input", "meta", "br", "hr", "link", "source"}
        opened: list[str] = []
        mismatched: list[str] = []

        class Balanced(HTMLParser):
            def handle_starttag(self, tag: str, attrs: object) -> None:
                if tag not in void:
                    opened.append(tag)

            def handle_endtag(self, tag: str) -> None:
                if tag in void:
                    return
                if not opened or opened[-1] != tag:
                    mismatched.append(tag)
                else:
                    opened.pop()

        parser = Balanced()
        parser.feed(page(view(correct=False, scores=(Score("iou", 0.5),)), view(correct=True)))

        assert (opened, mismatched) == ([], [])
