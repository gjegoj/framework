"""The test stage's headline numbers, put where a backend shows a run at a glance."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import lightning as L

from src.callbacks.registry import callback_registry
from src.core import Stage
from src.tracking import MetricKey, RecordsSummary

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

log = logging.getLogger(__name__)

SUMMARY_DECIMALS = 3
"""A summary table is read rather than computed with: 0.795 reads at a glance, 0.7948718070983887 does not.

Here rather than at each backend, because every table a run keeps shows the same reading, and a number
that differed between two of them would be the same measurement under two answers.
"""


@callback_registry.register("metric_summary")
class MetricSummary(L.Callback):
    """After the test stage, its headline numbers go to every backend that keeps such a table.

    Once a run is over its final numbers sit spread across as many lines as it drew, and what a
    reader wants then is one screen of them. Headline means the readings that stand for something
    whole — the objective, each number a task was measured by, a family at its mean — and never the
    classes below one: twenty leaves is not a glance.

    A backend without such a table is left alone: its numbers are in its lines either way, and this
    only adds the at-a-glance view where one exists. Rank is not checked here — a backend guards its
    own reporting, which is where that rule already lives.
    """

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # Every backend the run declared, rather than `trainer.logger`, which is only the first.
        tables = [one for one in trainer.loggers if isinstance(one, RecordsSummary)]
        if not tables:
            log.debug("Nothing this run records to keeps a summary table; its headline numbers stay in the lines.")
            return
        for name, value in headlines(trainer.callback_metrics, Stage.TEST).items():
            for table in tables:
                table.record_summary(name, value)


def headlines(logged: Mapping[str, Tensor], stage: Stage) -> dict[str, float]:
    """One stage's headline readings, named the way every stage names the same measurement.

    A free function because the selection is a rule about keys, and a rule about keys is worth
    reading and testing without a trainer to run it through.
    """
    found: dict[str, float] = {}
    for key, value in logged.items():
        headline = MetricKey.headline(key)
        if headline is not None and headline.stage is stage:
            found[headline.series] = round(float(value), SUMMARY_DECIMALS)
    return found
