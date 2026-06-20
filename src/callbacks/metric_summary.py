"""Metric-summary callback — report the test stage's headline metrics as single values.

After ``trainer.test`` finishes, push the main metrics to the logger's summary table
(e.g. ClearML "Single Values") so the final numbers are visible at a glance — scalars
and each vector metric's mean, without the per-class noise. A logger that is not a
``PlotLogger`` (e.g. the ``none`` logger) is simply skipped.
"""

from __future__ import annotations

import logging

import lightning as L

from src.core.enums import Stage
from src.core.ports import PlotLogger
from src.metrics.summary import summary_metrics

log = logging.getLogger(__name__)


class MetricSummaryCallback(L.Callback):
    """Report the test metrics to the logger's single-value summary table at ``on_test_end``."""

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        logger = trainer.logger
        if not isinstance(logger, PlotLogger):
            return
        values = summary_metrics(trainer.callback_metrics, Stage.TEST)
        for name, value in values.items():
            logger.log_single_value(name, value)
        if values:
            log.info("Reported %d test metrics to the logger summary table.", len(values))
