"""Dataset-distribution report: print tables and log plots before the first stage.

Clean split from the data layer: the ``DataModule`` *computes* the distributions
(``statistics()``); this callback *presents* them — a compact rich table per task in the
terminal and a histogram or box plot per task to the experiment logger. ``report_dataset_statistics``
is the pure, testable renderer; ``DatasetStatsCallback`` is the thin lifecycle glue that
fires it once, before training (or before eval in an eval-only run).
"""

from __future__ import annotations

import lightning as L

from src.reporting import report_dataset_statistics
from src.training.modules import LitDataModule


class DatasetStatsCallback(L.Callback):
    """Report dataset distributions once, before the first stage runs.

    Reads the distributions from the data module (``statistics()``) and renders them to the
    terminal plus the logger's plots. A no-op on non-zero ranks, when the data module
    cannot report statistics, or — for the plots — without a plot-capable logger.
    """

    def __init__(self) -> None:
        super().__init__()
        self._reported = False

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._report_once(trainer)

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._report_once(trainer)

    def _report_once(self, trainer: L.Trainer) -> None:
        # ``Trainer.datamodule`` is a public runtime attribute the type stubs do not expose.
        datamodule = getattr(trainer, "datamodule", None)
        if self._reported or not trainer.is_global_zero or not isinstance(datamodule, LitDataModule):
            return
        self._reported = True
        report_dataset_statistics(datamodule.statistics(), trainer.logger)
