"""The rendered page: the structure a browser, the sidebar, and the filter rely on."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pytest

from src.visualization import (
    Classification,
    Classifications,
    Image,
    SampleView,
    Score,
    Text,
    Verdict,
)
from src.visualization.fields import (
    _MIN_RIM_ALPHA,
    FieldContext,
    render_label,
    render_media,
)
from src.visualization.html import HtmlRenderer
from src.visualization.palette import hex_to_rgb
from tests.support.visualization_demo import demo_views


def rendered() -> str:
    return HtmlRenderer().render(demo_views(), title="samples/val")


@pytest.mark.parametrize(
    ("declared", "expected"),
    [({"max_side": 0}, "max_side >= 1"), ({"max_chip_chars": 0}, "max_chip_chars >= 1")],
)
def test_a_bound_that_could_only_draw_dots_is_refused_by_the_renderer(declared: dict[str, int], expected: str) -> None:
    """The samples callback refused these; anything building a page directly did not.

    This renderer is public and constructed straight from tests and from any consumer
    rendering a page of its own, and on that path ``max_side=0`` scaled every picture and
    every mask to a single pixel and uploaded a grid of dots without a word. A knob is
    refused by whoever owns it, not by whoever happens to pass it on.
    """
    with pytest.raises(ValueError, match=expected):
        HtmlRenderer(**declared)


def test_every_overlay_has_a_sidebar_row_with_the_same_key() -> None:
    """The checkbox toggles the overlay through their shared data-key; an orphan is dead UI."""
    page = rendered()

    for key in ("species::gt::cat", "species::pred::cat", "parts::gt::body", "tags::pred::studio"):
        assert page.count(f'data-key="{key}"') >= 2  # the overlay and its sidebar row


def test_a_task_named_with_an_underscore_renders_whole() -> None:
    """Structural field keys: the reference re-split glued strings and scrambled such tasks."""
    assert 'data-key="my_task::gt::ok"' in rendered()


def test_a_class_named_with_markup_does_not_become_markup() -> None:
    """The grid shows model output, and model output is not to be trusted as HTML."""
    page = rendered()

    assert "<script>alert" not in page
    assert "&lt;script&gt;alert" in page


def test_a_chip_is_shortened_on_the_cell_but_kept_whole_for_the_lightbox() -> None:
    """The lightbox swaps `data-full` back in, so truncation must not lose the text."""
    page = HtmlRenderer(max_chip_chars=8).render(demo_views(), title="t")

    assert 'data-full="&lt;script&gt;alert(1)&lt;/script&gt; 0.50"' in page  # what the lightbox reads
    assert ">&lt;script…<" in page  # the chip itself is cut


def test_cells_carry_their_verdicts_and_scores_for_the_filters() -> None:
    """Both filters read structured data off the cell; nothing re-parses a printed string."""
    page = rendered()

    assert "&quot;species&quot;: &quot;wrong&quot;" in page
    assert "&quot;age::mae&quot;: 1.2" in page  # keyed by task and metric, as the sliders are


def test_a_measured_number_is_printed_once_and_never_on_a_chip() -> None:
    """A draft put the error on the chip and in the verdict; the same number twice is noise."""
    page = rendered()

    assert page.count("mae 1.2") == 1  # the note, and nowhere else
    assert "Δ" not in page


def test_a_measured_task_gets_a_slider_and_a_judged_one_gets_no_control_of_its_own() -> None:
    """Correct/wrong cannot describe a mIoU; and per-task buttons crowded a 250px sidebar.

    Which task went wrong is read off the cell's own chips, or by switching one
    task's fields off in the tree above — the panel keeps only what nothing else
    can answer.
    """
    page = rendered()

    assert '<div class="filter range" data-key="parts::iou">' in page
    assert '<div class="filter range" data-key="parts::dice">' in page  # one task, two measures
    assert 'class="verdict"' not in page  # no per-task correct/wrong row survives


def test_a_slider_spans_the_scores_the_page_actually_holds() -> None:
    """A slider over a theoretical 0..1 would spend most of its travel where no sample is."""
    page = rendered()

    assert 'data-low="0.31" data-high="0.88"' in page  # the four IoUs on the page


def test_the_only_verdict_control_is_sample_wide_and_asks_for_whole_matches() -> None:
    """The question the grid exists for: a sample is correct only when every task matched."""
    page = rendered()

    assert page.count('class="sample-verdict"') == 3  # all / correct / mistakes, and nothing else
    assert 'value="mistakes"' in page


def test_a_hesitant_prediction_has_a_fainter_edge_than_a_confident_one() -> None:
    """Confidence as rim: hesitation is visible before the number is read.

    On the rim rather than the fill, because a translucent fill over an arbitrary
    image is hard to read whatever is behind it — the backing stays white so the
    number is legible at every confidence.
    """
    page = rendered()

    # Scoped to prediction chips: the stylesheet has rgba() colours of its own.
    rims = re.findall(r'class="layer chip pred"[^>]*?border-color:rgba\(\d+,\d+,\d+,([\d.]+)\)', page)
    alphas = sorted({float(value) for value in rims})

    assert len(alphas) > 1
    assert _MIN_RIM_ALPHA <= alphas[0] < alphas[-1] <= 1.0


def _rim_alpha(markup: str) -> float:
    found = re.search(r"border-color:rgba\(\d+,\d+,\d+,([\d.]+)\)", markup)
    assert found is not None
    return float(found.group(1))


def test_a_single_label_prediction_encodes_its_confidence_like_a_multilabel_one() -> None:
    """Multiclass is the commonest task there is, and its chip is a singular `Classification`.

    That renderer reached the same `_chip` as the multilabel one but never handed it
    the confidence, so it fell to the default that means *this label expressed none* —
    a 5% guess and a 99% one drew the same solid rim, while the multilabel chips
    beside them on the same page ramped correctly. Asserted here rather than on the
    demo page, which contains both kinds and so cannot tell them apart.
    """
    context = FieldContext("task", "pred", {"cat": "#3366cc"})

    faint = render_label(Classification("cat", confidence=0.05), context)[0].overlay
    solid = render_label(Classification("cat", confidence=0.99), context)[0].overlay

    assert _MIN_RIM_ALPHA <= _rim_alpha(faint) < _rim_alpha(solid) < 1.0


def test_a_slider_reaches_the_sample_at_its_own_bound() -> None:
    """The bounds and the cells have to round alike, or the extremes fall outside the band.

    The cells carried three decimals and the sliders full precision, so the lowest
    sample sat just under its own floor: the first drag of either handle dropped it,
    and the one sample the page exists to surface was the one guaranteed to vanish.
    """
    values = [0.5512345, 0.6, 0.7434999]
    views = [
        SampleView(
            media={"image": Image(pixels=np.zeros((4, 4, 3), dtype=np.uint8))},
            verdicts={"t": Verdict(scores=(Score(name="iou", value=value),))},
        )
        for value in values
    ]

    page = HtmlRenderer().render(views, title="t")

    bounds = re.search(r'class="edge low" value="([\d.]+)"[^>]*?max="([\d.]+)"', page)
    assert bounds is not None
    carried = sorted(float(value) for value in re.findall(r"&quot;t::iou&quot;: ([\d.]+)", page))
    assert float(bounds.group(1)) == carried[0]
    assert float(bounds.group(2)) == carried[-1]


def test_a_score_that_is_not_a_number_leaves_the_rest_of_the_page_working() -> None:
    """`json.dumps` writes a NaN as bare `NaN`, which `JSON.parse` rejects outright.

    One diverged metric would then take down every filter on the page at once — the
    verdict radio and every other task's slider included — with nothing on screen to
    say why. A diverged run is exactly when someone opens the samples grid.
    """
    view = SampleView(
        media={"image": Image(pixels=np.zeros((4, 4, 3), dtype=np.uint8))},
        verdicts={"t": Verdict(scores=(Score(name="mae", value=float("nan")),))},
    )

    page = HtmlRenderer().render([view], title="t")

    assert "NaN" not in page
    assert 'data-scores="{}"' in page
    assert 'class="filter range"' not in page  # no slider for a score that is not one
    assert "mae nan" in page  # but the note still says what the sample earned


def test_a_class_keeps_its_colour_when_the_page_shows_a_different_set() -> None:
    """A palette walks the hue circle in class order, so its seed must not be the page.

    Measured before the fix: `cat` was green on a two-class page and blue once a
    third class appeared, and `dog` went blue to red. Two epochs of one run then
    disagreed about what a class looks like — while a fixed `batch_index` exists
    precisely so the two pages can be compared.
    """
    vocabulary = {"label": ["bird", "cat", "dog"]}

    def page_for(shown: list[str]) -> str:
        view = SampleView(
            media={"image": Image(pixels=np.zeros((4, 4, 3), dtype=np.uint8))},
            fields={("label", "gt"): Classifications(tuple(Classification(name) for name in shown))},
        )
        return HtmlRenderer().render([view], title="t", classes=vocabulary)

    two, three = page_for(["cat", "dog"]), page_for(["bird", "cat", "dog"])

    colour = r'data-key="label::gt::cat"[^>]*?background:(#[0-9a-f]{6})'
    assert re.search(colour, two).group(1) == re.search(colour, three).group(1)  # type: ignore[union-attr]


def test_a_url_source_opens_and_a_local_path_copies() -> None:
    """A path in a browser is useless as a link and useful in a terminal — so it copies."""
    page = rendered()

    assert 'href="https://example.com/pets/cat_0001.jpg" target="_blank"' in page
    assert 'data-copy="/data/pets/images/dog_0042.jpg"' in page
    assert 'href="/data/pets/images/dog_0042.jpg"' not in page


def test_an_image_with_no_source_draws_no_pill_and_nothing_else_changes() -> None:
    """A run over in-memory arrays has no path; the cell must still be a cell."""
    import numpy as np

    from src.visualization import Image

    drawn = render_media(Image(pixels=np.zeros((4, 4, 3), dtype=np.uint8)), "image")

    assert 'class="src' not in drawn.markup
    assert drawn.markup.startswith('<img class="picture"')
    assert drawn.zone == "frame"


def test_the_choices_sit_on_a_line_of_their_own() -> None:
    """Options beside a name wrap mid-row in a 250px sidebar and stop lining up."""
    page = rendered()

    assert '<span class="title">samples</span><div class="options">' in page


def test_a_label_with_no_confidence_is_not_drawn_as_an_unconfident_one() -> None:
    """A regression chip has no confidence to encode, and the bottom of the ramp said it had none of it."""
    page = rendered()

    regression_chips = re.findall(
        r'class="layer chip pred"[^>]*?data-full="[\d.]+"[^>]*?border-color:rgba\(\d+,\d+,\d+,([\d.]+)\)',
        page,
    )

    assert regression_chips
    assert all(float(alpha) == 1.0 for alpha in regression_chips)


def test_a_prediction_is_written_in_its_class_colour_dark_enough_to_read() -> None:
    """Ground truth is the colour; a prediction is that colour as ink on white.

    The palette's own lightness fails contrast against white for most of a
    twelve-class palette, so the ink is the same hue re-emitted darker.
    """
    page = rendered()

    inks = {match for match in re.findall(r'class="layer chip pred"[^>]*?color:(#[0-9a-f]{6})', page)}

    assert inks
    assert all(_contrast_on_white(colour) >= 4.5 for colour in inks)


def test_a_sample_draws_every_input_it_has_not_only_the_first() -> None:
    """A CLIP-style run is an image beside a caption; drawing one of them halves the run."""
    assert "a tabby cat asleep on a sunlit windowsill" in rendered()


def test_the_page_fetches_nothing_from_anywhere() -> None:
    """ClearML embeds the page as-is: an external request would render it blank there."""
    page = rendered()

    assert "<link" not in page
    assert "<script src" not in page
    assert re.findall(r'src="(?!data:)[^"]*"', page) == []
    assert "<style>" in page and "<script>" in page


def test_an_unknown_label_names_the_kinds_that_are_known() -> None:
    """A new Label member must fail at its first render, not draw an empty cell."""

    @dataclass(frozen=True)
    class Detections:
        boxes: tuple[int, ...] = ()

    with pytest.raises(TypeError, match="Detections.*Classification"):
        render_label(Detections(), FieldContext(task="t", kind="gt", colors={}))  # type: ignore[arg-type]


def test_an_unknown_medium_names_the_kinds_that_are_known() -> None:
    """The same rule on the input side: a new modality is a loud gap, not a blank cell."""

    @dataclass(frozen=True)
    class Audio:
        samples: tuple[int, ...] = ()

    with pytest.raises(TypeError, match="Audio.*Image, Text"):
        render_media(Audio(), "sound")  # type: ignore[arg-type]


def test_a_missed_sample_says_so_on_the_cell_in_words_as_well_as_colour() -> None:
    """The page must answer "where is this wrong?" before anything is clicked.

    Words as well as colour, because colour alone is unreadable to a good share
    of the people who need the answer.
    """
    page = rendered()

    assert '<div class="badge bad">✗ wrong</div>' in page
    assert '<div class="badge ok">✓ correct</div>' in page
    assert page.count('class="cell bad"') == 3  # two wrong, and one with a task each way


def test_a_sample_that_missed_only_one_of_its_tasks_says_which_share_matched() -> None:
    """It is what the whole-match rule means: one wrong task makes the sample a mistake."""
    assert '<div class="badge bad">✗ 1/2 matched</div>' in rendered()


def test_the_chips_are_split_into_a_named_truth_row_and_prediction_row() -> None:
    """Filled against outlined is easy to lose at 11px on the same class in the same colour."""
    page = rendered()

    assert '<div class="chips"><span class="kind">gt</span>' in page
    assert '<div class="chips"><span class="kind">pred</span>' in page
    assert page.index('<span class="kind">gt</span>') < page.index('<span class="kind">pred</span>')


def test_the_page_counts_what_it_holds_and_what_each_filter_would_show() -> None:
    """A filter is read before it is clicked, and a page is read before it is filtered."""
    page = rendered()

    # Eleven cells: three carry a mistake, two are correct throughout, and the six
    # scored-but-unjudged ones are neither — which is what the counts must show.
    assert "11 samples · 3 with mistakes · 2 correct" in page
    assert 'mistakes <span class="tally">(3)</span>' in page
    assert 'correct <span class="tally">(2)</span>' in page
    assert 'all <span class="tally">(11)</span>' in page


def test_a_page_that_filters_to_nothing_offers_the_way_back() -> None:
    """An empty grid reads as a broken page unless it says otherwise."""
    page = rendered()

    assert 'id="empty"' in page
    assert 'id="reset"' in page
    assert "No sample matches the current filters." in page


def test_every_branch_of_the_tree_is_reachable_by_keyboard() -> None:
    """A clickable span is a mouse-only control; a button is not."""
    page = rendered()

    assert '<button class="caret" type="button" aria-expanded="false"' in page
    assert 'aria-label="Toggle species"' in page


def test_no_string_the_ir_accepts_can_reach_the_page_as_markup() -> None:
    """One blanket guard, so a new field cannot be forgotten at one call site.

    Escaping is applied by hand here rather than by a template engine, and the
    risk of hand-applied escaping is exactly that: one place, once, missed.
    """
    hostile = "<img src=x onerror=alert(1)>"
    view = SampleView(
        media={
            hostile: Image(pixels=np.zeros((4, 4, 3), dtype=np.uint8), source=hostile),
            "caption": Text(text=hostile),
        }
    )
    view.fields[(hostile, "gt")] = Classification(label=hostile)
    view.fields[(hostile, "pred")] = Classifications(classifications=(Classification(hostile, 0.5),))
    view.verdicts[hostile] = Verdict(correct=False, scores=(Score(name=hostile, value=1.0),))

    page = HtmlRenderer().render([view], title=hostile)

    assert hostile not in page
    assert "&lt;img src=x onerror=alert(1)&gt;" in page


def _contrast_on_white(colour: str) -> float:
    """WCAG contrast against white: 4.5 is the floor for 11px bold text."""

    def channel(value: float) -> float:
        value /= 255
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = (channel(component) for component in hex_to_rgb(colour))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return (1.0 + 0.05) / (luminance + 0.05)


def test_the_frame_takes_the_pictures_own_shape_so_overlays_land_on_it() -> None:
    """A square cell cropped the picture and stretched its masks; they disagreed.

    Measured before the fix: a 256x128 input showed columns 64..192 while its mask
    stretched over all 256, so a mask on the left quarter was drawn over the middle.
    Every future overlay — a box, a keypoint — inherits whichever this is.
    """
    page = rendered()

    assert 'style="aspect-ratio:2;width:100%"' in page  # the wide sample, 2:1
    assert 'style="aspect-ratio:1;width:100%"' in page  # the square ones


def test_a_portrait_frame_narrows_instead_of_stretching_the_picture() -> None:
    """The width is set with the ratio because a ratio alone does not survive the cell.

    A square cell gives the frame a definite width; a portrait ratio then derives a
    height taller than the cell, `max-height` clamps it back to a square, and the
    width never re-resolves — so the picture is stretched to fill it. The masks
    stretch identically, so nothing looks misaligned and the cell simply lies about
    the shape of its input.
    """
    tall = SampleView(media={"image": Image(pixels=np.zeros((256, 128, 3), dtype=np.uint8))})

    page = HtmlRenderer().render([tall], title="t")

    assert 'style="aspect-ratio:0.5;width:50%"' in page


def test_a_cell_with_nothing_to_show_still_has_a_frame() -> None:
    """A text-only sample has no geometry to take, and must not collapse the cell."""
    view = SampleView(media={"caption": Text(text="a cat")})

    assert 'style="aspect-ratio:1;width:100%"' in HtmlRenderer().render([view], title="t")
