"""The run's headline numbers, put where a backend shows a run at a glance rather than over time."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import lightning as L
import pytest
import torch
from lightning.pytorch.loggers import Logger
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from src.callbacks.metric_summary import MetricSummary, headlines
from src.core import Stage
from src.tracking import MetricKey, RecordsSummary


class Recording(Logger):
    """A backend with a summary table, which is the whole of what this callback asks for."""

    def __init__(self) -> None:
        super().__init__()
        self.summary: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "recording"

    @property
    def version(self) -> str:
        return "0"

    def log_hyperparams(self, *args: Any, **kwargs: Any) -> None:
        return None

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        return None

    def record_summary(self, name: str, value: float) -> None:
        self.summary[name] = value


class Plain(Logger):
    """A backend with no summary table: it keeps its numbers and is asked for nothing else."""

    @property
    def name(self) -> str:
        return "plain"

    @property
    def version(self) -> str:
        return "0"

    def log_hyperparams(self, *args: Any, **kwargs: Any) -> None:
        return None

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        return None


class Measured(L.LightningModule):
    """A run that logs one of each shape a report produces: a total, a number, a family of them."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def training_step(self, batch: Sequence[torch.Tensor], index: int) -> torch.Tensor:
        loss: torch.Tensor = self.layer(batch[0]).mean()
        self.log(str(MetricKey(Stage.TRAIN, "loss")), loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch: Sequence[torch.Tensor], index: int) -> None:
        return None

    def on_test_epoch_end(self) -> None:
        self.log(str(MetricKey(Stage.TEST, "loss")), 0.25)
        self.log(str(MetricKey(Stage.TEST, "f1/mean", task="label")), 0.75)
        self.log(str(MetricKey(Stage.TEST, "f1/cat", task="label")), 0.6)
        self.log(str(MetricKey(Stage.TEST, "f1/dog", task="label")), 0.9)

    def configure_optimizers(self) -> SGD:
        return SGD(self.parameters(), lr=0.1)


def rows() -> DataLoader[tuple[torch.Tensor, ...]]:
    return DataLoader(TensorDataset(torch.ones(4, 1)), batch_size=2)


def tested(*loggers: Logger) -> None:
    L.Trainer(
        callbacks=[MetricSummary()],
        logger=list(loggers) or False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator="cpu",
    ).test(Measured(), dataloaders=rows())


def test_a_backend_with_a_summary_table_says_so_by_having_the_method() -> None:
    assert isinstance(Recording(), RecordsSummary)
    assert not isinstance(Plain(), RecordsSummary)


def test_the_headline_numbers_of_the_test_stage_reach_the_summary() -> None:
    recording = Recording()

    tested(recording)

    assert recording.summary == pytest.approx({"loss": 0.25, "label/f1": 0.75})


def test_a_family_is_summarised_at_its_mean_and_not_class_by_class() -> None:
    """A summary table is read at a glance, and twenty class leaves is not a glance."""
    recording = Recording()

    tested(recording)

    assert "label/f1/cat" not in recording.summary


def test_every_backend_that_can_is_told_not_only_the_first() -> None:
    one, two = Recording(), Recording()

    tested(one, two)

    assert one.summary == two.summary != {}


def test_a_backend_without_a_summary_table_is_left_alone() -> None:
    """The numbers are in its logs either way; this only adds the at-a-glance view where there is one."""
    tested(Plain())


def test_a_run_that_records_nowhere_still_finishes() -> None:
    """`tracker: none` is the framework's own default, and a run under it has no backend at all — not
    one that keeps no table. Reading `trainer.logger`, which is the first of them and is `None` here,
    would end every such run on an attribute error at the last hook it runs."""
    tested()


def test_only_the_stage_asked_for_is_summarised() -> None:
    """A summary is about the stage it belongs to; a run that still held a validation reading would
    otherwise put two numbers under one name. Read here rather than through a trainer, because a
    trainer empties what it logged between the stages and so can never pose the question."""
    logged = {
        "test/loss": torch.tensor(0.25),
        # A reading of its own, not the test stage's under another name: a series drops the stage, so
        # two stages of one measurement would collide into the right answer and prove nothing.
        "val/label/f1/mean": torch.tensor(0.5),
    }

    assert headlines(logged, Stage.TEST) == pytest.approx({"loss": 0.25})
