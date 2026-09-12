"""The bar that also shows the numbers: what has been measured, where it is going, and how far.

The history and the table are plain Python and are tested as such. The one thing that needs a run is
the hookup — that the table really ends up inside the bar's own live region — and that is tested by
letting a real fit write to a captured screen and reading what a reader would have seen.
"""

from __future__ import annotations

from collections.abc import Sequence
from io import StringIO

import lightning as L
import pytest
import torch
from rich.console import Console
from rich.table import Table
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from src.callbacks.progress import MetricHistory, MetricsProgressBar, table
from src.core import Stage
from src.tracking import MetricKey

ACCURACY = MetricKey(Stage.VAL, "accuracy", task="label")
LOSS = MetricKey(Stage.TRAIN, "loss")


def rendered(built: Table) -> str:
    """What a reader sees, which is the only thing a table promises."""
    screen = StringIO()
    Console(file=screen, width=200).print(built)
    return screen.getvalue()


class Measured(L.LightningModule):
    """All the bar asks of a module: numbers under the run's grammar, and which way each is better."""

    def __init__(self, accuracies: Sequence[float] = (0.5, 0.8)) -> None:
        super().__init__()
        self.layer = nn.Linear(1, 1)
        self.accuracies = accuracies

    def metric_directions(self) -> dict[str, bool | None]:
        return {"label/accuracy": True}

    def training_step(self, batch: Sequence[torch.Tensor], index: int) -> torch.Tensor:
        loss: torch.Tensor = self.layer(batch[0]).mean()
        self.log(str(LOSS), loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Sequence[torch.Tensor], index: int) -> None:
        return None

    def on_validation_epoch_end(self) -> None:
        self.log(str(ACCURACY), self.accuracies[min(self.current_epoch, len(self.accuracies) - 1)])

    def on_train_epoch_end(self) -> None:
        self.log(str(MetricKey(Stage.TRAIN, "accuracy", task="label")), 0.6)

    def test_step(self, batch: Sequence[torch.Tensor], index: int) -> None:
        return None

    def on_test_epoch_end(self) -> None:
        self.log(str(MetricKey(Stage.TEST, "accuracy", task="label")), 0.75)

    def configure_optimizers(self) -> SGD:
        return SGD(self.parameters(), lr=0.1)


def rows() -> DataLoader[tuple[torch.Tensor, ...]]:
    return DataLoader(TensorDataset(torch.ones(4, 1)), batch_size=2)


class TestHistory:
    def test_it_holds_what_a_measurement_last_said(self) -> None:
        history = MetricHistory()

        history.observe(ACCURACY, 0.5)

        assert history.current[ACCURACY] == 0.5

    def test_a_measurement_that_moved_says_how_far(self) -> None:
        history = MetricHistory()

        history.observe(ACCURACY, 0.5)
        history.observe(ACCURACY, 0.8)

        assert history.moves[ACCURACY] == pytest.approx(0.3)

    @pytest.mark.parametrize(
        ("declared", "best"),
        [pytest.param(True, 0.8, id="higher is better"), pytest.param(False, 0.5, id="lower is better")],
    )
    def test_the_best_of_a_measurement_is_the_best_by_its_own_direction(self, declared: bool, best: float) -> None:
        history = MetricHistory({ACCURACY.series: declared})

        history.observe(ACCURACY, 0.5)
        history.observe(ACCURACY, 0.8)

        assert history.best[ACCURACY] == best

    def test_a_measurement_with_no_better_direction_keeps_no_best(self) -> None:
        """A confusion matrix is not improved, it is read; a best of one would be a number that lies."""
        history = MetricHistory({ACCURACY.series: None})

        history.observe(ACCURACY, 0.5)

        assert ACCURACY not in history.best

    def test_what_nobody_declares_is_a_loss(self) -> None:
        """The objective is logged by the loop itself and measured by no metric, so it declares nothing."""
        history = MetricHistory({ACCURACY.series: True})

        history.observe(LOSS, 0.9)
        history.observe(LOSS, 0.4)

        assert history.best[LOSS] == 0.4

    def test_a_second_declaration_adds_to_the_first_rather_than_replacing_it(self) -> None:
        """`setup` runs again for the test stage, and a module may answer for a different set of tasks
        there; replacing would leave what the fit measured with no direction and a backwards arrow."""
        history = MetricHistory({ACCURACY.series: True})

        history.declare({LOSS.series: False})

        assert history.better_higher(ACCURACY) is True
        assert history.better_higher(LOSS) is False


class TestTable:
    def test_every_stage_of_one_measurement_shares_a_row(self) -> None:
        history = MetricHistory({ACCURACY.series: True})
        history.observe(ACCURACY, 0.5)
        history.observe(MetricKey(Stage.TEST, "accuracy", task="label"), 0.75)

        text = rendered(table(history))

        assert "label/accuracy" in text
        assert text.count("label/accuracy") == 1, "one series is one row, whatever measured it"
        assert "0.5000" in text and "0.7500" in text

    def test_a_measurement_that_moved_shows_which_way(self) -> None:
        history = MetricHistory({ACCURACY.series: True})
        history.observe(ACCURACY, 0.5)
        history.observe(ACCURACY, 0.8)

        text = rendered(table(history))

        assert "▲" in text and "0.3000" in text

    def test_the_stages_are_named_as_columns_and_only_repeated_ones_carry_a_best(self) -> None:
        """Test is one pass after the fit: a best of a single reading is that reading, said twice."""
        text = rendered(table(MetricHistory()))

        assert "Best (train)" in text and "Best (val)" in text
        assert "Best (test)" not in text


class TestTheBar:
    def test_it_draws_the_table_under_the_bar_while_the_run_goes(self) -> None:
        screen = StringIO()
        bar = MetricsProgressBar(console_kwargs={"file": screen, "force_terminal": True, "width": 200})
        L.Trainer(max_epochs=1, callbacks=[bar], logger=False, enable_checkpointing=False, accelerator="cpu").fit(
            Measured(), train_dataloaders=rows(), val_dataloaders=rows()
        )

        assert "label/accuracy" in screen.getvalue()

    def test_it_takes_the_direction_of_each_measurement_from_the_module(self) -> None:
        """Declared `higher_is_better`, never guessed from a name: a guess shows the wrong best."""
        bar = MetricsProgressBar(console_kwargs={"file": StringIO(), "width": 200})
        L.Trainer(max_epochs=2, callbacks=[bar], logger=False, enable_checkpointing=False, accelerator="cpu").fit(
            Measured(accuracies=(0.8, 0.5)), train_dataloaders=rows(), val_dataloaders=rows()
        )

        assert bar.history.best[ACCURACY] == pytest.approx(0.8), "an accuracy that fell kept the better epoch"

    def test_the_test_reading_is_taken_in_after_the_numbers_that_make_it(self) -> None:
        """Nothing refreshes once a test epoch has ended, so the last column would arrive to an empty table."""
        bar = MetricsProgressBar(console_kwargs={"file": StringIO(), "width": 200})
        L.Trainer(callbacks=[bar], logger=False, enable_checkpointing=False, accelerator="cpu").test(
            Measured(), dataloaders=rows()
        )

        assert bar.history.current[MetricKey(Stage.TEST, "accuracy", task="label")] == pytest.approx(0.75)

    def test_the_last_epochs_readings_are_taken_in_before_the_display_stops(self) -> None:
        """A module reports its epoch after every callback has seen the end of one, so a stage's last
        reading is not logged yet when the last hook of that stage runs — and the fit ends on it."""
        bar = MetricsProgressBar(console_kwargs={"file": StringIO(), "width": 200})
        L.Trainer(max_epochs=1, callbacks=[bar], logger=False, enable_checkpointing=False, accelerator="cpu").fit(
            Measured(), train_dataloaders=rows(), val_dataloaders=rows()
        )

        assert bar.history.current[MetricKey(Stage.TRAIN, "accuracy", task="label")] == pytest.approx(0.6)

    def test_what_the_fit_showed_is_still_there_when_the_test_column_arrives(self) -> None:
        """Lightning empties its own metrics between the two, and those columns are what test is read against."""
        bar = MetricsProgressBar(console_kwargs={"file": StringIO(), "width": 200})
        run = L.Trainer(max_epochs=1, callbacks=[bar], logger=False, enable_checkpointing=False, accelerator="cpu")
        measured = Measured()
        run.fit(measured, train_dataloaders=rows(), val_dataloaders=rows())

        run.test(measured, dataloaders=rows())

        shown = rendered(table(bar.history))
        assert "0.5000" in shown and "0.7500" in shown

    def test_only_the_rows_a_run_asked_to_see(self) -> None:
        bar = MetricsProgressBar(metric_filters=["accuracy"], console_kwargs={"file": StringIO(), "width": 200})
        L.Trainer(max_epochs=1, callbacks=[bar], logger=False, enable_checkpointing=False, accelerator="cpu").fit(
            Measured(), train_dataloaders=rows(), val_dataloaders=rows()
        )

        assert ACCURACY in bar.history.current and LOSS not in bar.history.current
