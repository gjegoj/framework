"""ClearML behind Lightning's ``Logger`` and the artifact ports."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from io import StringIO
from typing import TYPE_CHECKING, Any

from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from src.core import log_keys
from src.loggers.registry import logger_registry

if TYPE_CHECKING:
    from argparse import Namespace

    from clearml.logger import Logger as ClearMLBackendLogger

    from src.core.entities import Bars, Curve, Matrix, Spread

log = logging.getLogger(__name__)

_DISPLAY_DECIMALS = 3
"""Matrix cells are read, not computed with; 0.333 reads, 0.3333333 does not."""


def _worth_showing(tags: Sequence[str | None]) -> list[str]:
    """The tags that say something, in the order declared and without repeats.

    A tag is written as an interpolation, so an unset group leaves an empty
    string behind (``${oc.select:scheduler.name,\'\'}`` with no scheduler), and the
    architecture is absent when a family cannot name itself. Neither is worth a
    chip in an experiment list.
    """
    return list(dict.fromkeys(tag for tag in tags if tag))


@logger_registry.register("clearml")
class ClearMLLogger(Logger):
    """One ClearML task carrying scalars, matrices, and curves of a run.

    Scalars arrive through Lightning's ``self.log`` path; matrices and curves
    through the structural artifact ports — one backend task holds them all,
    so nothing needs pairing. Keys are split by ``log_keys``, the grammar's
    owner: the stage becomes the series, so train/val/test of one value share
    a graph, losses and metrics alike.

    Declares only the names it must own; every other ``Task.init`` knob
    forwards verbatim, so any upstream option stays reachable from config.
    ``clearml`` imports lazily — the framework runs without it installed.

    **Every reporting method is ``rank_zero_only``, including the artifact ports.**
    The guard belongs here rather than with the callers: this is the object that
    knows there is a remote service behind it, and it is the one place a new
    consumer cannot forget. Callers may guard as well, and several do — but for
    their own reason, to skip building an artifact that would go nowhere, which is
    a different question from whether it may be sent. Left to the callers alone,
    `report_metric` runs on every rank at epoch end, so an unguarded curve uploaded
    one copy of itself per device.

    Parameters:
        project_name (str | None): ClearML project; backend default when None.
        task_name (str | None): Run name; backend default when None.
        tags (list[str] | None): Tags on the run. One that resolves to nothing is
            dropped: a tag is written as an interpolation (``${oc.select:adapters.name,''}``)
            and an unset group leaves an empty string, which a tracker should not show.
        architecture (str | None): Offered by assembly, not written in config —
            the key naming an architecture differs per backbone family and a
            composite backbone has none, so the model is asked and the answer
            joins the tags.
        reuse_last_task_id (bool): Declared for its framework default — a fresh
            run per fit beats ClearML's own reuse heuristic — and overridable
            like any other knob.
    """

    def __init__(
        self,
        project_name: str | None = None,
        task_name: str | None = None,
        tags: list[str] | None = None,
        architecture: str | None = None,
        reuse_last_task_id: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        from clearml import Task

        self._task: Task = Task.init(
            project_name=project_name,
            task_name=task_name,
            tags=_worth_showing([*(tags or []), architecture]),
            reuse_last_task_id=reuse_last_task_id,
            **kwargs,
        )
        self._backend: ClearMLBackendLogger = self._task.get_logger()

    @property
    def name(self) -> str:
        return str(self._task.name)

    @property
    def version(self) -> str:
        return str(self._task.id)

    @property
    def experiment(self) -> Any:
        return self._backend

    @rank_zero_only
    def log_hyperparams(self, params: Mapping[str, Any] | Namespace, *args: Any, **kwargs: Any) -> None:
        resolved = vars(params) if not isinstance(params, Mapping) else dict(params)
        self._task.connect(resolved)

    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        for key, value in metrics.items():
            title, series = log_keys.split_for_tracker(key)
            self._backend.report_scalar(title=title, series=series, value=float(value), iteration=step or 0)

    @rank_zero_only
    def log_matrix(self, title: str, matrix: Matrix, iteration: int) -> None:
        rounded = matrix.value.detach().cpu().float().round(decimals=_DISPLAY_DECIMALS).numpy()
        labels = list(matrix.labels) if matrix.labels is not None else None
        self._backend.report_confusion_matrix(
            title=title,
            series="matrix",
            matrix=rounded,
            iteration=iteration,
            xlabels=labels,
            ylabels=labels,
            xaxis=matrix.xaxis,
            yaxis=matrix.yaxis,
        )

    @rank_zero_only
    def log_spread(self, title: str, spread: Spread, iteration: int) -> None:
        """One box per stage, built from the summary rather than from the values.

        ClearML has no native box plot, so this is the one artifact the framework
        draws itself — plotly accepts a box described entirely by its quartiles, so
        no column has to be carried into memory to draw one. Imported inside the
        method, as ``log_curve`` imports numpy, so a CLI start does not pay for a
        plotting library it will not use.

        The whiskers are the observed extremes, not Tukey fences; ``Spread`` says so.
        """
        import plotly.graph_objects as go

        figure = go.Figure()
        for series, box in zip(spread.series, spread.boxes, strict=True):
            figure.add_trace(
                go.Box(
                    name=series,
                    # The series is also its position on the axis. Without one every
                    # box is drawn at zero, so the stages stack on top of each other
                    # and the axis carries a bare "0" instead of their names.
                    x=[series],
                    q1=[box.q25],
                    median=[box.median],
                    q3=[box.q75],
                    lowerfence=[box.minimum],
                    upperfence=[box.maximum],
                    mean=[box.mean],
                    boxmean=True,
                )
            )
        figure.update_layout(title=title, xaxis_title=spread.xaxis, yaxis_title=spread.yaxis)
        self._backend.report_plotly(title=title, series=spread.yaxis, figure=figure, iteration=iteration)

    @rank_zero_only
    def log_bars(self, title: str, bars: Bars, iteration: int) -> None:
        """One grouped bar chart: a series per stage, a bar per class.

        ``mode="group"`` rather than the stacked default, because the question a
        class balance answers is how the *splits* compare on one class — stacking
        would put that comparison inside a single column.
        """
        for series, values in zip(bars.series, bars.values, strict=True):
            self._backend.report_histogram(
                title=title,
                series=series,
                values=list(values),
                iteration=iteration,
                xlabels=list(bars.labels),
                xaxis=bars.xaxis,
                yaxis=bars.yaxis,
                mode="group",
            )

    @rank_zero_only
    def log_curve(self, title: str, curve: Curve, iteration: int) -> None:
        import numpy as np

        if curve.series is None:
            raise ValueError(
                f"Curve '{title}' arrived without series names; a curve reaches a backend "
                "completed — the router (core.reporting) fills them from the task's classes."
            )
        for series, x, y in zip(curve.series, curve.x, curve.y, strict=True):
            scatter = np.column_stack([x.detach().cpu().float().numpy(), y.detach().cpu().float().numpy()])
            self._backend.report_scatter2d(
                title=title,
                series=series,
                iteration=iteration,
                scatter=scatter,
                mode="lines",
                xaxis=curve.xaxis,
                yaxis=curve.yaxis,
            )

    @rank_zero_only
    def log_single_value(self, name: str, value: float) -> None:
        self._backend.report_single_value(name, round(float(value), _DISPLAY_DECIMALS))

    @rank_zero_only
    def log_html(self, title: str, html: str, iteration: int) -> None:
        """Ship a self-contained page as media, which is what ClearML embeds in place.

        ``report_media`` with an ``html`` extension is the one call that renders
        a page inside the Debug Samples panel; an artifact upload would give a
        file to download instead, and nobody downloads a file to look at a batch.
        The page carries its own CSS and JS, so nothing is fetched when it opens.
        """
        self._backend.report_media(
            title=title,
            series="grid",
            iteration=iteration,
            stream=StringIO(html),
            file_extension="html",
        )

    @rank_zero_only
    def finalize(self, status: str) -> None:
        try:
            self._task.flush()
        except Exception as error:  # noqa: BLE001 — telemetry must not take the run's results with it
            log.warning("ClearML flush failed during finalize: %s", error)
