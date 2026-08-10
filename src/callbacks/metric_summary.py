"""The test stage's headline numbers, pushed to the tracker's summary table."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import lightning as L

from src.core import log_keys
from src.core.reporting import SingleValueLogger
from src.core.taxonomy import Stage

if TYPE_CHECKING:
    from collections.abc import Mapping

log = logging.getLogger(__name__)


def headline_metrics(metrics: Mapping[str, Any], stage: str) -> dict[str, float]:
    """The stage's headline numbers, keyed without the stage prefix.

    A free function so the selection is testable without Lightning. Scalars
    pass as they are; a vector metric contributes its ``mean`` under the
    collapsed name; per-class leaves and other stages drop.
    """
    selected: dict[str, float] = {}
    for key, value in metrics.items():
        prefix, _, name = key.partition(log_keys.SEPARATOR)
        if prefix != stage or not name:
            continue
        segments = name.split(log_keys.SEPARATOR)
        if len(segments) >= 3:
            if segments[-1] != log_keys.MEAN:
                continue  # a vector's per-class leaf: noise at summary altitude
            name = log_keys.SEPARATOR.join(segments[:-1])
        selected[name] = float(value)
    return selected


class MetricSummary(L.Callback):
    """Report the test metrics to the tracker's single-value table at ``on_test_end``.

    After ``trainer.test`` the final numbers sit spread across scalar log keys, and a
    backend with a summary table (ClearML's "Single Values") shows them at a glance.
    Headline means aggregates — the loss, each scalar metric, and a vector metric's
    ``mean`` — never its per-class leaves.

    A backend without a summary table is skipped quietly: the numbers are still in the
    logs and the plots, and this only adds the at-a-glance view where one exists.
    """

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        # Every configured backend, not `trainer.logger`: that is the first of them, so a
        # run with two trackers used to fill one summary table and leave the other empty.
        summaries = [one for one in trainer.loggers if isinstance(one, SingleValueLogger)]
        if not summaries:
            log.debug("No configured backend has a summary table; test headline metrics stay in the logs.")
            return
        reported = headline_metrics(trainer.callback_metrics, Stage.TEST)
        for backend in summaries:
            for name, value in reported.items():
                backend.log_single_value(name, value)
        if reported:
            log.info("Reported %d test metrics to %d summary table(s).", len(reported), len(summaries))
