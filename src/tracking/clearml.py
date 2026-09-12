"""ClearML behind Lightning's logger, and behind the one drawing port it can serve.

The service is reached lazily twice over: the client is imported only when a run is actually started
there, so a run that never declares `tracker: clearml` does not need it installed, and the run itself
is started by the first thing reported rather than by the constructor. Every reporting method is
rank-zero only, the drawing one included — this is the object that knows there is a remote service
behind it, and unguarded, a matrix would be uploaded once per device.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from io import StringIO
from typing import TYPE_CHECKING, Any, cast

from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.logger import rank_zero_experiment
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from src.tracking.keys import SEGMENT, MetricKey
from src.tracking.registry import tracker_registry

if TYPE_CHECKING:
    from argparse import Namespace

    from clearml import Task
    from clearml.logger import Logger as Backend

    from src.core import Bars, Matrix

log = logging.getLogger(__name__)

MATRIX_DECIMALS = 3
"""Matrix cells are read rather than computed with: 0.333 reads, 0.3333333 does not."""

PAGE = "grid"
"""What a page is filed under inside its title — this service files every medium under a series too."""

DEFAULT_LINE = "value"
"""The line a value that names none of its own is drawn as — `epoch` is a graph with one series."""


@tracker_registry.register("clearml")
class ClearMLTracker(Logger):
    """One ClearML task carrying a run's numbers and the readings that mean an image.

    Parameters:
        project_name: ClearML project; the service's own default when None.
        task_name: The run's name there; the service's own default when None.
        tags: Chips the experiment list filters by; one that resolved to nothing is dropped.
        reuse_last_task_id: A fresh run per fit beats the service's own reuse heuristic.
        **options: Forwarded to ``Task.init`` verbatim, so every upstream knob stays reachable.
    """

    def __init__(
        self,
        project_name: str | None = None,
        task_name: str | None = None,
        tags: Sequence[str] | None = None,
        reuse_last_task_id: bool = False,
        **options: Any,
    ) -> None:
        super().__init__()
        self._declared: dict[str, Any] = {
            "project_name": project_name,
            "task_name": task_name,
            "tags": _worth_showing(tags or ()),
            "reuse_last_task_id": reuse_last_task_id,
            **options,
        }
        self._task: Task | None = None

    @property
    @rank_zero_experiment
    def experiment(self) -> Task:
        """The run on the service, made by the first thing reported to it and only where reporting happens.

        A constructor runs on every device, so starting the run there would start one per device, all
        but the first empty. ``rank_zero_experiment`` is Lightning's own answer: every other device is
        handed a stand-in that quietly does nothing, which is exactly what a follower should report.
        """
        if self._task is None:
            from clearml import Task

            self._task = Task.init(**self._declared)
        return self._task

    @property
    def _reporter(self) -> Backend:
        """Where numbers and images go. The service holds one per run, so this holds none."""
        # Cast, because Lightning's rank-zero guard around `experiment` is untyped by construction:
        # what it hands a follower is a stand-in, not a run.
        return cast("Backend", self.experiment.get_logger())

    @property
    def name(self) -> str:
        """What the service calls this run. Asking for the identity starts the run, as it does in Lightning's
        own remote loggers; a device that starts none has no name to give."""
        _ = self.experiment
        return "" if self._task is None else str(self._task.name)

    @property
    def version(self) -> str:
        _ = self.experiment
        return "" if self._task is None else str(self._task.id)

    @rank_zero_only
    def log_hyperparams(self, params: Mapping[str, Any] | Namespace, *args: Any, **kwargs: Any) -> None:
        self.experiment.connect(dict(params) if isinstance(params, Mapping) else vars(params))

    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        reporter = self._reporter
        for key, value in metrics.items():
            title, series = _drawn_as(key)
            reporter.report_scalar(title=title, series=series, value=float(value), iteration=step or 0)

    @rank_zero_only
    def log_matrix(self, title: str, matrix: Matrix, iteration: int) -> None:
        labels = list(matrix.labels) if matrix.labels is not None else None
        self._reporter.report_confusion_matrix(
            title=title,
            series=DEFAULT_LINE,
            matrix=matrix.value.detach().cpu().float().round(decimals=MATRIX_DECIMALS).numpy(),
            iteration=iteration,
            xlabels=labels,
            ylabels=labels,
            xaxis=matrix.xaxis,
            yaxis=matrix.yaxis,
        )

    @rank_zero_only
    def log_bars(self, title: str, bars: Bars, iteration: int) -> None:
        """The ``DrawsBars`` port: one grouped bar chart, a series per split and a bar per class.

        Grouped rather than the stacked default, because the question a class balance answers is how
        the *splits* compare on one class — stacking puts that comparison inside a single column.
        """
        for series, values in zip(bars.series, bars.values, strict=True):
            self._reporter.report_histogram(
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
    def log_html(self, title: str, html: str, iteration: int) -> None:
        """The ``ShowsPage`` port: a page shipped as media, which is what this service renders in place.

        ``report_media`` with an html extension is the one call that opens a page inside the debug
        samples panel; uploading it as an artifact would give a file to download instead, and nobody
        downloads a file to look at a batch. The page carries its own styling, so nothing is fetched
        when it opens — which a panel that forbids a second request is the reason for.
        """
        self._reporter.report_media(
            title=title, series=PAGE, iteration=iteration, stream=StringIO(html), file_extension="html"
        )

    @rank_zero_only
    def record_summary(self, name: str, value: float) -> None:
        """The ``RecordsSummary`` port: a number with no iteration axis, in the table kept for those."""
        self._reporter.report_single_value(name=name, value=value)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        """A run that reported nothing has nothing to push, and starting one to say so would be a lie."""
        if self._task is None:
            return
        try:
            self._task.flush()
        except Exception as error:
            log.warning("ClearML could not be reached while finishing the run: %s", error)


def _drawn_as(key: str) -> tuple[str, str]:
    """A logged key as this service draws it: one graph per title, one line per series.

    - ``train/species/cross_entropy`` → ``("species/cross_entropy", "train")``: the stages of one
      number are the comparison a reader makes, so they belong on one graph.
    - ``val/species/f1/cat`` → ``("val/species/f1", "cat")``: a family's own leaves are the comparison
      instead, at the cost of train and val sitting apart.
    - ``lr/backbone`` → ``("lr", "backbone")``: no stage at all, so the leaves are again the comparison.
    """
    head, _, rest = key.partition(SEGMENT)
    if not rest:
        return key, DEFAULT_LINE
    try:
        parsed = MetricKey.parse(key)
    except ValueError:
        return head, rest
    return (parsed.family, parsed.leaf) if SEGMENT in parsed.name else (rest, head)


def _worth_showing(tags: Sequence[str]) -> list[str]:
    """The tags that say something, in the order declared and without repeats.

    A tag is written as an interpolation, so a group that is off leaves an empty string behind
    (``${oc.select:scheduler.name,''}`` with no scheduler); an empty chip filters nothing.
    """
    return list(dict.fromkeys(tag for tag in tags if tag))
