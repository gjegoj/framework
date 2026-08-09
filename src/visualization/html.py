"""``SampleView``s → one self-contained HTML page: grid, sidebar, lightbox, filters."""

from __future__ import annotations

import json
import math
from importlib.resources import files
from typing import TYPE_CHECKING

from src.visualization.entities import KINDS
from src.visualization.fields import (
    MAX_CHIP_CHARS,
    ZONES,
    FieldContext,
    FieldItem,
    MediaItem,
    attr,
    field_prefix,
    leaves_of,
    number,
    render_label,
    render_media,
    text,
)
from src.visualization.palette import task_palette

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from src.visualization.entities import Kind, SampleView

_ASSETS = files("src.visualization") / "assets"

MAX_DISPLAY_SIDE = 256
"""How many pixels of a picture or mask reach the page, on its longest side.

Bounds what the page weighs, not what the tensor holds. Measured: eight cells of
a ten-class segmentation inlined whole at 512px make a 49 MB page in 13 seconds,
every N epochs, into a tracker that then has to embed it. Cells display at about
230px, so this costs nothing there and only softens the lightbox. Public because
the callback offers it as a knob and both must mean the same number.
"""

_LIGHTBOX = """<div class="lb hidden" id="lb"><div class="stage" id="lb-stage">
  <div id="lb-holder"></div>
  <button class="nav prev" id="lb-prev" title="Previous (←)" aria-label="Previous">‹</button>
  <button class="nav next" id="lb-next" title="Next (→)" aria-label="Next">›</button>
  <button class="close" id="lb-close" title="Close (Esc)" aria-label="Close">✕</button>
  <div class="count" id="lb-count"></div>
</div></div>"""

_EMPTY = """<div class="empty hidden" id="empty">
  <p>No sample matches the current filters.</p>
  <button type="button" id="reset">Reset filters</button>
</div>"""


class HtmlRenderer:
    """A layout shell over ``fields.render_label`` and ``fields.render_media``.

    No plotting library and no template engine. This assembles the page and never asks
    what kind of thing it is drawing — what one label or one input looks like is
    ``fields.py``'s business, decided by type.

    The page answers one question before anything is clicked: *where does this model get
    it wrong?* A cell that missed is outlined and carries a badge saying so in words as
    well as in colour, the filters count what they would show, and a combination that
    hides everything says so and offers a way back.

    Assets are read once through ``importlib.resources``, so they survive being
    installed as a wheel, and inlined — a page in a tracker makes no second request.

    Parameters:
        max_chip_chars (int): Chip text budget before truncation; the lightbox
            swaps the full text back in.
        max_side (int | None): Bound every inlined picture and mask to this many
            pixels on its longest side. ``None`` inlines them whole, which is an
            opt-out a segmentation run should think twice about — see
            ``MAX_DISPLAY_SIDE``.
    """

    def __init__(self, max_chip_chars: int = MAX_CHIP_CHARS, max_side: int | None = MAX_DISPLAY_SIDE) -> None:
        self._max_chip_chars = max_chip_chars
        self._max_side = max_side
        self._css = (_ASSETS / "grid.css").read_text(encoding="utf-8")
        self._script = (_ASSETS / "grid.js").read_text(encoding="utf-8")

    def render(
        self,
        views: Sequence[SampleView],
        title: str,
        classes: Mapping[str, Sequence[str]] | None = None,
    ) -> str:
        """One page showing every view, with the controls that narrow it.

        ``classes`` is each task's whole vocabulary, which anchors its palette so
        two pages of one run colour a class the same way — see ``_palettes``.
        """
        palettes = _palettes(views, classes or {})
        cells: list[str] = []
        items: list[FieldItem] = []
        for view in views:
            cell, cell_items = self._cell(view, palettes)
            cells.append(cell)
            items.extend(cell_items)
        grid = "\n".join(cells)
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
            for (task, kind), label in sorted(view.fields.items())
            for item in render_label(label, self._context(task, kind, palettes))
        ]
        shown = [render_media(media, alias, self._max_side) for alias, media in view.media.items()]
        drawn = [(item.zone, item.overlay) for item in items] + [(one.zone, one.markup) for one in shown]
        zones = {
            zone: "".join(markup for at, markup in drawn if at == zone)
            for zone in ZONES
            # Chips are the one zone not placed flat: `_chip_rows` splits them into a
            # truth row and a prediction row, which needs each piece's `kind`.
            if zone != "chips"
        }
        cell = (
            f'<div class="cell {_standing(view)}" data-verdicts="{attr(json.dumps(_verdicts(view)))}"'
            f' data-scores="{attr(json.dumps(_scores(view)))}">'
            # The frame is the cell's coordinate system: it takes the picture's own
            # shape, and the picture and every overlay fill it. Anything positioned
            # in percentages of it — a mask now, a box or a keypoint later — lands on
            # the pixels it is explaining, whatever shape the model was trained at.
            f'<div class="frame" style="{_frame_style(shown)}">'
            f'{zones["frame"]}<div class="cover">{zones["cover"]}</div></div>'
            f'<div class="stack">{_chip_rows(items)}{zones["caption"]}</div>'
            f"{_badge(view)}{_note(view)}</div>"
        )
        return cell, items

    def _context(self, task: str, kind: Kind, palettes: Mapping[str, Mapping[str, str]]) -> FieldContext:
        return FieldContext(task, kind, palettes.get(task, {}), self._max_chip_chars, self._max_side)


def _aspect(shown: Sequence[MediaItem]) -> float:
    """The shape the frame takes: the first medium that has one, or a square."""
    for medium in shown:
        if medium.aspect is not None:
            return medium.aspect
    return 1.0


def _frame_style(shown: Sequence[MediaItem]) -> str:
    """The frame's shape, and the width that lets it keep that shape inside a square cell.

    ``aspect-ratio`` alone does not survive: with a definite ``width: 100%`` a
    portrait frame derives a height taller than the cell, ``max-height`` clamps it
    back to a square, and the width is never re-resolved — so the picture is
    stretched into the square instead of letterboxed, and the masks stretch with it
    so nothing looks wrong. The cell is square, so a frame of aspect ``a`` fits at
    ``min(1, a)`` of its width, and both orientations keep their proportions.
    """
    aspect = _aspect(shown)
    return f"aspect-ratio:{aspect:.6g};width:{100 * min(1.0, aspect):.4g}%"


def _palettes(views: Sequence[SampleView], classes: Mapping[str, Sequence[str]]) -> dict[str, dict[str, str]]:
    """One palette per task, anchored to the task's whole vocabulary where it is known.

    A palette walks the hue circle in class order, so seeding it from the classes
    that happen to be on *this* page moves every colour the moment a prediction
    introduces or drops one — measured, ``cat`` went from green to blue when a
    third class appeared, and ``dog`` from blue to red. Two pages of one run would
    then disagree about what a class looks like, which is what a fixed
    ``batch_index`` exists to make comparable. A task that never declared class
    names falls back to the leaves the page shows.
    """
    present: dict[str, set[str]] = {}
    for view in views:
        for (task, _), label in view.fields.items():
            present.setdefault(task, set()).update(leaves_of(label))
    return {task: task_palette(task, sorted(leaves | set(classes.get(task, ())))) for task, leaves in present.items()}


def _chip_rows(items: Sequence[FieldItem]) -> str:
    """Chips split into a truth row and a prediction row, each named.

    A filled chip against an outlined one is the whole difference otherwise, and
    at 11px, on the same class, in the same colour, that is easy to lose. The
    split also fixes the reading order: gt then pred, whatever order the
    annotators happened to write their fields in.
    """
    rows = []
    for kind in KINDS:
        chips = "".join(item.overlay for item in items if item.zone == "chips" and item.kind == kind)
        if chips:
            rows.append(f'<div class="chips"><span class="kind">{text(kind)}</span>{chips}</div>')
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
    """The verdict on the cell itself, in words as well as in colour.

    Colour alone leaves the answer unreadable to a good share of the people who
    need it, and the mixed case has no colour anyway: ``1/2 matched`` is what
    "this sample is a mistake" means when only one of its tasks missed.
    """
    matched, judged = _matched(view)
    if not judged:
        return ""
    if matched == judged:
        return '<div class="badge ok">✓ correct</div>'
    said = "wrong" if judged == 1 else f"{matched}/{judged} matched"
    return f'<div class="badge bad">✗ {text(said)}</div>'


def _verdicts(view: SampleView) -> dict[str, str]:
    """What the sample filter reads off the cell; tasks with no binary verdict stay out of it."""
    return {
        task: ("correct" if verdict.correct else "wrong")
        for task, verdict in view.verdicts.items()
        if verdict.correct is not None
    }


def score_key(task: str, name: str) -> str:
    """A slider's identity: which task measured, and which way. Glued here, never re-split."""
    return f"{task}::{name}"


_SCORE_DECIMALS = 3
"""How precisely a score reaches the page — the cell's attribute and the slider's bounds alike.

They must round the same way, and did not: cells carried three decimals while the
sliders carried full precision, so narrowing a band made the lowest-scoring sample
fail its own floor by a rounding error and disappear — the one sample the sliders
exist to find. Rounded in ``_measurements``, which is the only reader of a score.
"""


def _measurements(view: SampleView) -> list[tuple[str, str, float]]:
    """Every score this sample earned, as the page will carry it: task, name, value.

    A non-finite score is dropped rather than written. ``json.dumps`` spells a NaN
    as bare ``NaN``, which ``JSON.parse`` rejects — so one diverged metric would
    take down every filter on the page at once, not just its own slider. The note
    still prints it, because a NaN is a fact about the run worth seeing.
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
    measured = [f"{score.name} {number(score.value)}" for verdict in view.verdicts.values() for score in verdict.scores]
    return f'<div class="note">{text(" · ".join(measured))}</div>' if measured else ""


def _tally(views: Sequence[SampleView]) -> tuple[int, int, int]:
    """Samples on the page, those with every task matched, and those with any task missed."""
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

    Keyed by both, so a task that measures itself two ways gets two sliders, and one
    that measures itself once is the same code with one entry.

    The range comes from the page rather than from theory: an IoU column where every
    sample sits between 0.55 and 0.71 gives a slider with resolution where the
    samples are, not one whose useful travel is a tenth of its length.
    """
    seen: dict[tuple[str, str], list[float]] = {}
    for view in views:
        for task, name, value in _measurements(view):
            seen.setdefault((task, name), []).append(value)
    return {measured: (min(values), max(values)) for measured, values in sorted(seen.items())}


def _sidebar(items: Sequence[FieldItem]) -> str:
    """Task → kind → leaf, each level a checkbox over the level below."""
    rows: dict[str, dict[Kind, dict[str, FieldItem]]] = {}
    for item in items:
        rows.setdefault(item.task, {}).setdefault(item.kind, {})[item.key] = item
    return "".join(_task_node(task, rows[task]) for task in sorted(rows))


def _task_node(task: str, kinds: Mapping[Kind, Mapping[str, FieldItem]]) -> str:
    children = "".join(_kind_node(task, kind, kinds[kind]) for kind in KINDS if kind in kinds)
    return _node(title=task, prefix=field_prefix(task), children=children, css_class="task")


def _kind_node(task: str, kind: Kind, leaves: Mapping[str, FieldItem]) -> str:
    children = "".join(_leaf_row(item) for item in sorted(leaves.values(), key=lambda entry: entry.leaf))
    return _node(title=kind, prefix=field_prefix(task, kind), children=children, css_class="kind")


def _node(title: str, prefix: str, children: str, css_class: str) -> str:
    """A collapsible branch. The caret is a button, so a keyboard reaches it like a mouse does."""
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
        if item.kind == "gt"
        else f'<span class="swatch" style="border:2px solid {item.color}"></span>'
    )
    return (
        f'<label class="row"><input type="checkbox" class="cls" data-key="{attr(item.key)}" checked>'
        f"{swatch}{text(item.leaf)}</label>"
    )


def _filters(views: Sequence[SampleView]) -> str:
    """The question the grid exists to answer, plus one dial per measured score.

    Only two controls, and deliberately so. *Show me the mistakes* is what anyone
    opens the page for, and a sample counts as correct only when every judged task
    on it matched. A row of correct/wrong buttons per task stood here too; it
    answered the narrower "which task went wrong", crowded the panel in a 250px
    sidebar, and was reachable anyway — the field checkboxes above already isolate
    one task's chips and masks. What no button can answer is which samples scored
    badly and how badly, so that is what the sliders are for. Both narrow the same
    set and combine with AND.
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


_SLIDER_STEPS = 100
"""Enough travel to pick out a tail, few enough that a drag lands somewhere round."""


def _slider(task: str, name: str, low: float, high: float) -> str:
    """A two-handle range over one score: pick a band, not a threshold with a guessed direction.

    Both ends move because which end is *bad* depends on the score — a low mIoU
    and a high error are the same complaint — and a band also answers "show me
    the middle", which no single threshold can.
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
