"""The dataset report: one table and one chart per task, dispatched by the shape itself."""

from __future__ import annotations

import pytest
from rich.console import Console

from src.callbacks.dataset_summary import draw, table_for
from src.callbacks.registry import callback_registry
from src.console import HEADER_STYLE, TITLE_STYLE
from src.core import Stage
from src.core.entities import Bars, ClassDistribution, Distribution, Spread, ValueDistribution


def drawn(table: object) -> str:
    console = Console(width=200, record=True, force_terminal=False)
    console.print(table)
    return console.export_text()


def balance() -> dict[Stage, Distribution]:
    return {
        Stage.TRAIN: ClassDistribution(counts={"cat": 30, "dog": 10, "rare": 0}),
        Stage.VAL: ClassDistribution(counts={"cat": 6, "dog": 4, "rare": 0}),
    }


def spread() -> dict[Stage, Distribution]:
    return {
        Stage.TRAIN: ValueDistribution(
            count=3, mean=2.0, deviation=1.0, minimum=1.0, q25=1.5, median=2.0, q75=2.5, maximum=3.0
        )
    }


class OnlyBars:
    """A backend that draws bars and nothing else — so a spread sent here would be skipped."""

    def __init__(self) -> None:
        self.bars: list[Bars] = []

    def log_bars(self, title: str, bars: Bars, iteration: int) -> None:
        self.bars.append(bars)


class OnlySpread:
    """And one that draws only boxes. Separate stands, because a backend implementing
    both cannot show *which* port a shape was narrowed on — the call would land either way."""

    def __init__(self) -> None:
        self.spreads: list[Spread] = []

    def log_spread(self, title: str, spread: Spread, iteration: int) -> None:
        self.spreads.append(spread)


class Deaf:
    """A backend implementing neither — the CSV case, and it must not be an error."""

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...


def test_the_table_carries_the_stage_size_rather_than_letting_it_be_inferred() -> None:
    """Only a single-label column has as many counts as rows.

    A multilabel one counts every label a row carries and a mask counts pixels, so a
    reader adding up a column to learn the stage's size would be wrong by a factor
    nothing on the page reveals. The total is carried in and printed.
    """
    shown = drawn(table_for("label", balance(), {Stage.TRAIN: 40, Stage.VAL: 10}))

    assert "Total" in shown
    assert "40" in shown and "10" in shown


def test_a_class_no_split_produced_is_marked_not_left_among_the_numbers() -> None:
    """It is the row worth reading, and a bare 0 in a column of numbers is easy to miss."""
    shown = drawn(table_for("label", balance(), {Stage.TRAIN: 40, Stage.VAL: 10}))

    assert "rare" in shown
    assert "0 (0.0%)" in shown


def test_the_stages_are_read_off_the_enum_and_never_re_listed() -> None:
    """Train before val before test, from `Stage` itself, so a new member needs no edit here."""
    reversed_order = {Stage.VAL: balance()[Stage.VAL], Stage.TRAIN: balance()[Stage.TRAIN]}

    shown = drawn(table_for("label", reversed_order, {}))

    assert shown.index("Train") < shown.index("Val")


def test_a_column_with_nothing_missing_shows_one_plain_number() -> None:
    """Rows held and rows holding a number are the same until a value is missing.

    Two columns for that would repeat each other on every row of an ordinary
    column, and a column that repeats its neighbour is noise.
    """
    complete = {Stage.TRAIN: spread()[Stage.TRAIN]}

    shown = drawn(table_for("age", complete, {Stage.TRAIN: 3}))

    assert "value spread" in shown
    assert "missing" not in shown


def test_a_missing_value_is_named_in_the_same_column_rather_than_left_to_subtraction() -> None:
    """Where the two do differ, that difference is the whole point of having counted."""
    shown = drawn(table_for("age", spread(), {Stage.TRAIN: 5}))

    assert "5 (2 missing)" in shown


def test_a_class_balance_reaches_the_bars_port_and_not_the_other_one() -> None:
    """One series per stage, so the splits stand side by side on each class.

    Asserted against two single-port backends: one implementing both would receive
    the call whichever port the registration narrowed on, and could not tell them
    apart.
    """
    drawing, boxes = OnlyBars(), OnlySpread()

    draw(balance()[Stage.TRAIN], "dataset/label", balance(), [drawing, boxes])

    (bars,) = drawing.bars
    assert not boxes.spreads
    assert bars.series == ("train", "val")
    assert bars.labels == ("cat", "dog", "rare")
    assert bars.values == ((30.0, 10.0, 0.0), (6.0, 4.0, 0.0))


def test_a_value_spread_reaches_the_spread_port_carrying_the_summary_itself() -> None:
    """The box *is* the five-number summary, so `Spread` holds it rather than a copy.

    The reference kept a parallel `BoxStats` whose docstring admitted it mirrored
    the distribution field for field — two records to keep in step by hand.
    """
    bars, boxes = OnlyBars(), OnlySpread()

    draw(spread()[Stage.TRAIN], "dataset/age", spread(), [bars, boxes])

    (drawing,) = boxes.spreads
    assert not bars.bars
    assert drawing.series == ("train",)
    assert drawing.boxes == (spread()[Stage.TRAIN],)


def test_a_tracker_that_draws_neither_is_simply_not_asked() -> None:
    """A CSV run keeps its scalars; a dataset report is not worth failing a run over."""
    draw(balance()[Stage.TRAIN], "dataset/label", balance(), [Deaf()])
    draw(spread()[Stage.TRAIN], "dataset/age", spread(), [Deaf()])


def test_a_shape_with_no_table_is_refused_by_name() -> None:
    """A new distribution cannot be half-supported: the table side says so at once."""
    with pytest.raises(TypeError, match="No table for str"):
        table_for("t", {Stage.TRAIN: "not a distribution"}, {})  # type: ignore[dict-item]


def test_a_shape_with_no_chart_is_refused_by_name() -> None:
    """And so does the chart side, which is the half that would otherwise fail silently.

    A shape that gained a table but no chart would print and then send nothing at
    all — the failure mode dispatch was chosen to prevent.
    """
    with pytest.raises(TypeError, match="No chart for str"):
        draw("not a distribution", "t", {}, [])  # type: ignore[arg-type]


def test_both_tables_are_dressed_like_every_other_table_this_framework_prints() -> None:
    """Three callbacks print a table and a reader meets them seconds apart.

    Named once in `callbacks.console` and read from there — including by the
    progress bar, which had the colour spelled inline — so "consistent" is a fact
    about the code rather than a coincidence between three string literals.
    """
    balance_table = table_for("label", balance(), {Stage.TRAIN: 40})
    spread_table = table_for("age", spread(), {Stage.TRAIN: 5})

    for table in (balance_table, spread_table):
        assert table.header_style == HEADER_STYLE
        assert table.title_style == TITLE_STYLE
        headers = [str(column.header) for column in table.columns]
        # Only the first letter, so a header like "25%" is left alone and one like
        # "FLOPs" would keep its own casing.
        assert headers == [header[:1].upper() + header[1:] for header in headers]


def test_it_is_reachable_from_config_by_name() -> None:
    assert "dataset_summary" in callback_registry
