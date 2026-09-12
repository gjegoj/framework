"""Views into one self-contained page: a grid, a sidebar that switches layers, and filters that narrow it.

No template engine and no plotting library — what one label looks like is ``renderers.py``'s business,
and this is the shell around it. The page answers one question, *where does this model get it wrong*:
a cell that missed is outlined and badged, and every filter counts what it would leave.
"""

from __future__ import annotations

import json
import math
from importlib.resources import files
from typing import TYPE_CHECKING

from src.visualization.entities import SIDES, SampleView
from src.visualization.palette import task_palette
from src.visualization.png import data_uri
from src.visualization.renderers import (
    FieldContext,
    FieldItem,
    attr,
    field_prefix,
    leaves_of,
    number,
    render_label,
    score_key,
    source_pill,
    text,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from src.visualization.entities import Side

_ASSETS = files(__package__) / "assets"

MAX_DISPLAY_SIDE = 256
"""How many pixels of an image or a mask reach the page, on its longest side.

This bounds what the page weighs, not what the model saw. A cell inlines its image and one layer per
class per side, so the weight is cells times layers times this — at 512 a segmentation grid runs to
tens of megabytes, into a tracker that then has to embed it. Cells display at about 230px, so the cost
lands only on the lightbox.
"""

_LIGHTBOX = """<div class="lb hidden" id="lb"><div class="stage" id="lb-stage">
  <div id="lb-holder"></div>
  <button class="nav prev" id="lb-prev" title="Previous (left arrow)" aria-label="Previous">‹</button>
  <button class="nav next" id="lb-next" title="Next (right arrow)" aria-label="Next">›</button>
  <button class="close" id="lb-close" title="Close (Esc)" aria-label="Close">✕</button>
  <div class="count" id="lb-count"></div>
</div></div>"""

_EMPTY = """<div class="empty hidden" id="empty">
  <p>No sample matches the current filters.</p>
  <button type="button" id="reset">Reset filters</button>
</div>"""

_SCORE_DECIMALS = 3
"""How precisely a score reaches the page — the cell's attribute and the slider's bounds alike.

They have to round the same way, and one rounding is what guarantees it: a cell rounded to three
decimals under a slider carrying full precision makes the lowest-scoring sample fail its own floor and
disappear, which is the one sample the sliders exist to find.
"""

_SLIDER_STEPS = 100
"""Enough travel to pick out a tail, few enough that a drag lands somewhere round."""


class HtmlRenderer:
    """Turns a page's worth of views into markup that carries everything it needs.

    Parameters:
        max_side: Bound every inlined image and mask to this many pixels on its longest side;
            ``None`` inlines them whole — see :data:`MAX_DISPLAY_SIDE`.
    """

    def __init__(self, max_side: int | None = MAX_DISPLAY_SIDE) -> None:
        # Checked here rather than by whoever passes it on: a knob is refused by whoever owns it, and
        # this class is public — a page rendered directly would otherwise take the value unchecked.
        # `None` rather than zero for "inline whole": at zero every image and every mask scales to
        # one pixel, and the page builds a grid of dots without a word.
        if max_side is not None and max_side < 1:
            raise ValueError(f"An image needs a pixel a side: needs max_side >= 1 or None, got {max_side}.")
        self._max_side = max_side
        self._css = (_ASSETS / "grid.css").read_text(encoding="utf-8")
        self._script = (_ASSETS / "grid.js").read_text(encoding="utf-8")

    def render(
        self, views: Sequence[SampleView], title: str, classes: Mapping[str, Sequence[str]] | None = None
    ) -> str:
        """One page showing every view, with the controls that narrow it.

        ``classes`` is each task's whole vocabulary, which anchors its palette so two pages of one run
        colour a class the same way — see :func:`_palettes`.
        """
        palettes = _palettes(views, classes or {})
        drawn = [self._cell(view, palettes) for view in views]
        items = [item for _, cell_items in drawn for item in cell_items]
        grid = "\n".join(cell for cell, _ in drawn)
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{text(title)}</title><style>{self._css}</style></head>\n"
            "<body>\n"
            f'<div class="sidebar"><h2>fields</h2>{_sidebar(items)}{_filters(views)}</div>\n'
            f'<div class="main"><h2>{text(title)}<span class="summary">{text(_summary(views))}</span></h2>'
            f'<div class="grid">{grid}</div>{_EMPTY}</div>\n'
            f"{_LIGHTBOX}\n"
            f"<script>{self._script}</script>\n"
            "</body></html>"
        )

    def _cell(self, view: SampleView, palettes: Mapping[str, Mapping[str, str]]) -> tuple[str, list[FieldItem]]:
        items = [
            item
            for (task, side), label in sorted(view.fields.items())
            for item in render_label(label, self._context(task, side, palettes))
        ]
        covers = "".join(item.overlay for item in items if item.zone == "cover")
        height, width = view.image.pixels.shape[:2]
        cell = (
            f'<div class="cell {_standing(view)}" data-verdicts="{attr(json.dumps(_verdicts(view)))}"'
            f' data-scores="{attr(json.dumps(_scores(view)))}">'
            # The frame is the cell's coordinate system: it takes the image's own shape, and every
            # overlay fills it. Anything placed in percentages of it lands on the pixels it explains,
            # whatever shape the model was trained at.
            f'<div class="frame" style="{_frame_style(width, height)}">'
            f'<img class="image" alt="sample" src="{data_uri(view.image.pixels, self._max_side)}">'
            f"{source_pill(view.image.source)}"
            f'<div class="cover">{covers}</div></div>'
            f'<div class="stack">{_chip_rows(items)}</div>'
            f"{_badge(view)}{_note(view)}</div>"
        )
        return cell, items

    def _context(self, task: str, side: Side, palettes: Mapping[str, Mapping[str, str]]) -> FieldContext:
        return FieldContext(task, side, palettes.get(task, {}), max_side=self._max_side)


def _frame_style(width: int, height: int) -> str:
    """The frame's shape, and the width that lets it keep that shape inside a square cell.

    ``aspect-ratio`` alone does not survive: with a definite ``width: 100%`` a portrait frame derives
    a height taller than the cell, ``max-height`` clamps it back to a square, and the width is never
    re-resolved — so the image is stretched into the square and the masks stretch with it, which
    makes nothing look wrong. The cell is square, so a frame of aspect ``a`` fits at ``min(1, a)`` of
    its width and both orientations keep their proportions.
    """
    aspect = width / height
    return f"aspect-ratio:{aspect:.6g};width:{100 * min(1.0, aspect):.4g}%"


def _palettes(views: Sequence[SampleView], classes: Mapping[str, Sequence[str]]) -> dict[str, dict[str, str]]:
    """One palette per task, anchored to the task's whole vocabulary where it is known.

    A palette walks the hue circle in class order, so anything that changes the set of names moves
    every colour after it — and two pages of one run would then disagree about what a class looks
    like, which is what a fixed batch exists to make comparable. The declared vocabulary alone is what
    seeds it, because that is the one set that does not depend on what a model happened to say: a
    prediction outside it draws in the unclaimed grey rather than shifting everything it sorts before.
    A task that declared no vocabulary has only the leaves this page shows to go on.
    """
    present: dict[str, set[str]] = {}
    for view in views:
        for (task, _), label in view.fields.items():
            present.setdefault(task, set()).update(leaves_of(label))
    return {
        task: task_palette(task, classes[task] if task in classes else sorted(leaves))
        for task, leaves in present.items()
    }


def _chip_rows(items: Sequence[FieldItem]) -> str:
    """Chips split into a truth row and a prediction row, each named.

    A filled chip against an outlined one is otherwise the whole difference, and at chip size, on one
    class, in one hue, that is easy to lose. The split also fixes the reading order whatever order the
    fields were written in.
    """
    rows = []
    for side in SIDES:
        chips = "".join(item.overlay for item in items if item.zone == "chips" and item.side == side)
        if chips:
            rows.append(f'<div class="chips"><span class="kind">{text(side)}</span>{chips}</div>')
    return "".join(rows)


def _matched(view: SampleView) -> tuple[int, int]:
    """How many of this sample's judged tasks matched, out of how many judged it."""
    judged = [verdict for verdict in view.verdicts.values() if verdict.correct is not None]
    return sum(1 for verdict in judged if verdict.correct), len(judged)


def _standing(view: SampleView) -> str:
    matched, judged = _matched(view)
    if not judged:
        return ""
    return "ok" if matched == judged else "bad"


def _badge(view: SampleView) -> str:
    """The verdict on the cell in words as well as in colour.

    Colour alone leaves the answer unreadable to a good share of the people who need it, and the mixed
    case has no colour anyway: ``1/2 matched`` is what "this sample is a mistake" means when only one
    of its tasks missed.
    """
    matched, judged = _matched(view)
    if not judged:
        return ""
    if matched == judged:
        return '<div class="badge ok">✓ correct</div>'
    reads = "wrong" if judged == 1 else f"{matched}/{judged} matched"
    return f'<div class="badge bad">✗ {text(reads)}</div>'


def _verdicts(view: SampleView) -> dict[str, str]:
    """What the sample filter reads off the cell; a task with no yes-or-no answer stays out of it."""
    return {
        task: ("correct" if verdict.correct else "wrong")
        for task, verdict in view.verdicts.items()
        if verdict.correct is not None
    }


def _measurements(view: SampleView) -> list[tuple[str, str, float]]:
    """Every score this sample earned, as the page carries it: task, name, value.

    A number that diverged is dropped rather than written. ``json.dumps`` spells a NaN as bare ``NaN``,
    which the page's parser rejects, so one of them would take down every filter at once and not only
    its own slider. The note still prints it, because a NaN is a fact about the run worth seeing.
    """
    return [
        (task, score.name, round(score.value, _SCORE_DECIMALS))
        for task, verdict in view.verdicts.items()
        for score in verdict.scores
        if math.isfinite(score.value)
    ]


def _scores(view: SampleView) -> dict[str, float]:
    """What the range sliders read off the cell — the same numbers the note prints."""
    return {score_key(task, name): value for task, name, value in _measurements(view)}


def _note(view: SampleView) -> str:
    """Every measured number this sample earned, printed once and only here."""
    printed = [f"{score.name} {number(score.value)}" for verdict in view.verdicts.values() for score in verdict.scores]
    return f'<div class="note">{text(" · ".join(printed))}</div>' if printed else ""


def _tally(views: Sequence[SampleView]) -> tuple[int, int, int]:
    """Samples on the page, those every task matched on, and those any task missed on."""
    standings = [_standing(view) for view in views]
    return len(views), standings.count("ok"), standings.count("bad")


def _summary(views: Sequence[SampleView]) -> str:
    """What the page holds, before any filter narrows it."""
    total, correct, mistakes = _tally(views)
    if not correct and not mistakes:
        return f" · {total} samples"
    return f" · {total} samples · {mistakes} with mistakes · {correct} correct"


def _spans(views: Sequence[SampleView]) -> dict[tuple[str, str], tuple[float, float]]:
    """Per measured thing — task *and* metric — the range the page actually covers.

    The range comes from the page rather than from theory: an overlap column where every sample sits
    between 0.55 and 0.71 gives a slider with its resolution where the samples are, rather than one
    whose useful travel is a tenth of its length.
    """
    seen: dict[tuple[str, str], list[float]] = {}
    for view in views:
        for task, name, value in _measurements(view):
            seen.setdefault((task, name), []).append(value)
    return {measured: (min(values), max(values)) for measured, values in sorted(seen.items())}


def _sidebar(items: Sequence[FieldItem]) -> str:
    """Task, then side, then leaf — each level a checkbox over the level below."""
    rows: dict[str, dict[Side, dict[str, FieldItem]]] = {}
    for item in items:
        rows.setdefault(item.task, {}).setdefault(item.side, {})[item.key] = item
    return "".join(_task_node(task, rows[task]) for task in sorted(rows))


def _task_node(task: str, sides: Mapping[Side, Mapping[str, FieldItem]]) -> str:
    children = "".join(_side_node(task, side, sides[side]) for side in SIDES if side in sides)
    return _node(title=task, prefix=field_prefix(task), children=children, css_class="task")


def _side_node(task: str, side: Side, leaves: Mapping[str, FieldItem]) -> str:
    children = "".join(_leaf_row(item) for item in sorted(leaves.values(), key=lambda entry: entry.leaf))
    return _node(title=side, prefix=field_prefix(task, side), children=children, css_class="kind")


def _node(title: str, prefix: str, children: str, css_class: str) -> str:
    """A collapsible branch. The caret is a button, so a keyboard reaches it the way a mouse does."""
    return (
        f'<div class="node {css_class}"><div class="header">'
        f'<button class="caret" type="button" aria-expanded="false" aria-label="Toggle {attr(title)}">▸</button>'
        f'<input type="checkbox" class="grp" data-prefix="{attr(prefix)}" checked>'
        f'<span class="title">{text(title)}</span></div>'
        f'<div class="children">{children}</div></div>'
    )


def _leaf_row(item: FieldItem) -> str:
    swatch = (
        f'<span class="swatch" style="background:{item.color}"></span>'
        if item.side == "gt"
        else f'<span class="swatch" style="border:2px solid {item.color}"></span>'
    )
    return (
        f'<label class="row"><input type="checkbox" class="cls" data-key="{attr(item.key)}" checked>'
        f"{swatch}{text(item.leaf)}</label>"
    )


def _filters(views: Sequence[SampleView]) -> str:
    """The question the grid exists to answer, plus one dial per measured score.

    *Show me the mistakes* is what anyone opens the page for — a sample counts as correct only when
    every judged task on it matched — and the sliders answer which samples scored badly and how badly.
    Both narrow the same set, and they combine.
    """
    spans = _spans(views)
    total, correct, mistakes = _tally(views)
    if not correct and not mistakes and not spans:
        return ""
    parts = ["<h2>show</h2>"]
    if correct or mistakes:
        parts.append(_sample_choice(total, correct, mistakes))
    parts.extend(_slider(task, name, low, high) for (task, name), (low, high) in spans.items())
    parts.append('<div class="shown" id="shown"></div>')
    return "".join(parts)


def _sample_choice(total: int, correct: int, mistakes: int) -> str:
    """The one verdict control, its options counted so the page is read before it is clicked."""
    counted = (("all", total), ("correct", correct), ("mistakes", mistakes))
    options = "".join(
        f'<label><input type="radio" name="sample-verdict" class="sample-verdict" '
        f'value="{value}"{" checked" if value == "all" else ""}>{value} '
        f'<span class="tally">({count})</span></label>'
        for value, count in counted
    )
    return f'<div class="filter"><span class="title">samples</span><div class="options">{options}</div></div>'


def _slider(task: str, name: str, low: float, high: float) -> str:
    """A two-handle band over one score: pick a range, not a threshold with a guessed direction.

    Both ends move because which end is *bad* depends on the score — a low overlap and a high error
    are the same complaint — and a band also answers "show me the middle", which no threshold can.
    """
    step = (high - low) / _SLIDER_STEPS if high > low else 1.0
    key = attr(score_key(task, name))
    said = attr(f"{task} {name}")
    handle = f'min="{low}" max="{high}" step="{step}" data-key="{key}" data-low="{low}" data-high="{high}"'
    return (
        f'<div class="filter range" data-key="{key}">'
        f'<span class="title">{text(task)}</span><span class="metric">{text(name)}</span>'
        f'<span class="bounds">{number(low)} – {number(high)}</span>'
        f'<div class="rail"><div class="fill"></div>'
        f'<input type="range" class="edge low" value="{low}" {handle} aria-label="{said} lower bound">'
        f'<input type="range" class="edge high" value="{high}" {handle} aria-label="{said} upper bound">'
        f"</div></div>"
    )
