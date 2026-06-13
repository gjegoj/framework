"""ClearML logger adapter — implements both Lightning's Logger and PlotLogger.

One object, one ClearML Task, two entry points:
- Scalars arrive via Lightning's ``self.log`` → ``Logger.log_metrics``.
- Matrices arrive via ``PlotLogger.log_matrix`` (called by MatrixMetricHandler).

Both paths share the same underlying ``ClearML Task``, so all metrics land in
one experiment run with no "pairing" between separate objects.

ClearML is an optional dependency: ``uv add clearml``. The class is defined
here but imported lazily (via ``src.loggers.registry``) so the rest of the
framework works without it installed.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping
from io import StringIO
from typing import TYPE_CHECKING, Any

import numpy as np
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from src.core.ports import PlotLogger

if TYPE_CHECKING:
    from clearml import Task
    from clearml.logger import Logger as ClearMLBackendLogger


class ClearMLLogger(Logger, PlotLogger):
    """Lightning + PlotLogger backed by ClearML.

    Inherits Lightning's ``Logger`` (for ``self.log`` scalar path) and our
    ``PlotLogger`` port (for ``MatrixMetricHandler`` matrix path). A single
    ClearML ``Task`` handles both.

    Metric name splitting: ``a/b/c`` → title ``a/b``, series ``c``.
    Single-component names → title ``<name>``, series ``"value"``.

    Parameters:
        project_name (str | None): ClearML project name (default: ClearML default).
        task_name (str | None): ClearML task/run name (default: script name).
        tags (list[str] | None): Optional task tags.
    """

    def __init__(
        self,
        project_name: str | None = None,
        task_name: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        Logger.__init__(self)
        # Lazy import so clearml is not required for the rest of the framework.
        from clearml import Task

        self._task: Task = Task.init(
            project_name=project_name,
            task_name=task_name,
            tags=tags,
            reuse_last_task_id=False,
        )
        self._clearml_logger: ClearMLBackendLogger = self._task.get_logger()

    # ---------------------------------------------------------------- Logger

    @property
    def name(self) -> str:
        return str(self._task.name)

    @property
    def version(self) -> str:
        return str(self._task.id)

    @property
    def experiment(self) -> Any:
        return self._clearml_logger

    @rank_zero_only
    def log_hyperparams(self, params: dict[str, Any] | Namespace, *args: Any, **kwargs: Any) -> None:
        params_dict = vars(params) if isinstance(params, Namespace) else dict(params)
        self._task.connect(params_dict)

    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        for name, value in metrics.items():
            title, series = self._split_metric_name(name)
            self._clearml_logger.report_scalar(
                title=title,
                series=series,
                value=float(value),
                iteration=step or 0,
            )

    @rank_zero_only
    def finalize(self, status: str) -> None:
        try:
            self._task.flush()
        except Exception:
            pass

    def close(self) -> None:
        """Flush and close the ClearML task (call once at end of training)."""
        try:
            self._task.flush()
            self._task.close()
        except Exception:
            pass

    # ----------------------------------------------------------- PlotLogger

    @rank_zero_only
    def log_matrix(
        self,
        title: str,
        matrix: Any,
        iteration: int,
        labels: list[str] | None = None,
        xaxis: str | None = None,
        yaxis: str | None = None,
    ) -> None:
        self._clearml_logger.report_confusion_matrix(
            title=title,
            series="matrix",
            matrix=_round_matrix(matrix),
            iteration=iteration,
            xlabels=labels,
            ylabels=labels,
            xaxis=xaxis,
            yaxis=yaxis,
        )

    @rank_zero_only
    def log_curve(
        self,
        title: str,
        x: Any,
        y: Any,
        iteration: int,
        series: str = "curve",
        xaxis: str | None = None,
        yaxis: str | None = None,
    ) -> None:
        scatter = np.column_stack([x.cpu().float().numpy(), y.cpu().float().numpy()])
        self._clearml_logger.report_scatter2d(
            title=title,
            series=series,
            iteration=iteration,
            scatter=scatter,
            mode="lines",
            xaxis=xaxis,
            yaxis=yaxis,
        )

    @rank_zero_only
    def log_html(self, title: str, html: str, iteration: int) -> None:
        self._clearml_logger.report_media(
            title=title,
            series="grid",
            iteration=iteration,
            stream=StringIO(html),
            file_extension="html",
        )

    # ---------------------------------------------------------------- utils

    @staticmethod
    def _split_metric_name(name: str) -> tuple[str, str]:
        """Split ``a/b/c`` → ``("a/b", "c")``; single name → ``(name, "value")``."""
        parts = name.split("/")
        if len(parts) == 1:
            return name, "value"
        return "/".join(parts[:-1]), parts[-1]


def _round_matrix(matrix: Any) -> np.ndarray:
    """Round a matrix to 3 decimals for display.

    A normalized confusion matrix is full of long floats (e.g. ``0.333333…``); 3
    decimals (``0.001`` precision) is plenty and keeps the ClearML cells readable.
    Integer (count) matrices are unaffected by rounding.
    """
    rounded: np.ndarray = np.round(matrix.cpu().float().numpy(), 3)
    return rounded
