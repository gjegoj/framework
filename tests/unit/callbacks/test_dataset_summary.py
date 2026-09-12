"""What the run is about to train on, said before anything runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import lightning as L
import pytest
from rich.console import Console

from src.callbacks.dataset_summary import DatasetSummary, bars_for, table_for
from src.core import Bars, ClassDistribution, Distribution, ValueDistribution
from tests.support.pages import NumbersOnly
from tests.unit.callbacks.conftest import prepared

BALANCE: dict[str, Distribution] = {
    "train": ClassDistribution({"cat": 3, "dog": 1, "bird": 0}),
    "val": ClassDistribution({"cat": 1, "dog": 1, "bird": 0}),
}

SPREAD: dict[str, Distribution] = {
    "train": ValueDistribution(
        count=4, mean=2.5, deviation=1.29, minimum=1.0, q25=1.75, median=2.5, q75=3.25, maximum=4.0
    ),
}


def printed(renderable: Any, *, styled: bool = False) -> str:
    shown = Console(width=200, record=True, force_terminal=styled)
    shown.print(renderable)
    return shown.export_text(styles=styled)


class TestClassBalance:
    def test_each_class_is_counted_and_shared_per_split(self) -> None:
        text = printed(table_for("species", BALANCE, rows={"train": 4, "val": 2}))

        assert "cat" in text and "3 (75.0%)" in text
        assert "Train" in text and "Val" in text

    def test_a_class_no_split_produced_is_marked_rather_than_left_as_a_zero_among_numbers(self) -> None:
        """It is the row worth reading: a vocabulary that names something the data never shows."""
        text = printed(table_for("species", BALANCE, rows={"train": 4, "val": 2}), styled=True)

        assert "\x1b[33mbird" in text
        assert "\x1b[33mcat" not in text

    def test_the_row_count_is_carried_in_rather_than_added_up_from_the_counts(self) -> None:
        """Only a single-label column has as many counts as rows. A multilabel one counts every label a
        row carries and a mask counts pixels, so a reader adding up the column would be wrong by a
        factor nothing on the table reveals — here eight labels over four rows."""
        text = printed(table_for("tags", {"train": ClassDistribution({"a": 6, "b": 2})}, rows={"train": 4}))

        (total,) = [line for line in text.splitlines() if "Total" in line]
        assert "4" in total and "8" not in total


class TestValueSpread:
    def test_the_five_numbers_and_the_two_that_go_with_them(self) -> None:
        text = printed(table_for("age", SPREAD, rows={"train": 4}))

        for header in ("Mean", "Std", "Min", "25%", "50%", "75%", "Max"):
            assert header in text

    def test_a_column_with_a_missing_cell_says_how_many_rather_than_hiding_it(self) -> None:
        missing = {"train": ValueDistribution(3, 2.0, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0)}

        assert "(1 missing)" in printed(table_for("age", missing, rows={"train": 4}))


class TestCharts:
    def test_a_balance_is_drawn_as_one_group_of_bars_per_split(self) -> None:
        drawn = bars_for(BALANCE)

        assert drawn == Bars(
            series=("train", "val"),
            values=((3.0, 1.0, 0.0), (1.0, 1.0, 0.0)),
            labels=("cat", "dog", "bird"),
            xaxis="class",
            yaxis="count",
        )

    def test_a_spread_is_left_to_the_table_it_already_fits_in(self) -> None:
        """Five numbers are five numbers; an image of them says less than the row does."""
        assert bars_for(SPREAD) is None


class TestTheRun:
    def test_it_says_what_a_real_run_is_about_to_train_on(
        self, declaration: Mapping[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        built = prepared(declaration, callbacks=[{"name": "dataset_summary"}])
        built.trainer.fit(built.module, datamodule=built.data)

        shown = capsys.readouterr().out
        assert "species" in shown and "cat" in shown

    def test_it_says_it_once_however_many_stages_run(
        self, declaration: Mapping[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        built = prepared(declaration, callbacks=[{"name": "dataset_summary"}])
        built.trainer.fit(built.module, datamodule=built.data)
        built.trainer.test(built.module, datamodule=built.data)

        assert capsys.readouterr().out.count("class balance") == 1

    def test_a_backend_that_draws_no_bars_keeps_its_numbers_and_the_table_is_still_printed(
        self, declaration: Mapping[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        built = prepared(
            declaration,
            tracker={"_target_": "tests.support.pages.NumbersOnly"},
            callbacks=[{"name": "dataset_summary"}],
        )
        built.trainer.fit(built.module, datamodule=built.data)

        assert any(isinstance(one, NumbersOnly) for one in built.trainer.loggers)
        assert "class balance" in capsys.readouterr().out


class TestDeclaration:
    def test_a_pipeline_this_framework_did_not_prepare_is_not_an_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A summary is a display: a run driven by somebody else's data module loses the table, not itself."""
        DatasetSummary().on_fit_start(L.Trainer(logger=False), L.LightningModule())

        assert "class balance" not in capsys.readouterr().out


class TestNothingIsHidden:
    def test_a_class_only_a_later_split_holds_still_has_a_row(self) -> None:
        """Taking the classes from the first split drops it from the table while its rows still count
        toward that split's total — the column then reads 11% and 11% and says nothing about the rest."""
        per_split: dict[str, Distribution] = {
            "train": ClassDistribution({"cat": 3, "dog": 1}),
            "val": ClassDistribution({"cat": 1, "dog": 1, "rogue": 7}),
        }

        text = printed(table_for("species", per_split, rows={"train": 4, "val": 9}))

        drawn = bars_for(per_split)
        assert "rogue" in text and "7 (77.8%)" in text
        assert drawn is not None and drawn.labels == ("cat", "dog", "rogue")


class TestAnEvaluationOnlyRun:
    def test_it_still_says_what_it_is_evaluating_on(
        self, declaration: Mapping[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A run that only tests reaches no fit hook, and a summary it never printed is a summary
        nobody asked for twice."""
        built = prepared(declaration, callbacks=[{"name": "dataset_summary"}])

        built.trainer.test(built.module, datamodule=built.data)

        assert "class balance" in capsys.readouterr().out
