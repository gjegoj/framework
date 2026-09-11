"""ClearML behind Lightning's logger, and behind the one drawing port it can serve.

The service is reached lazily: this module is imported whenever the package is, and a run that never
declares `tracker: clearml` must not need the client installed. Every reporting method is rank-zero
only, the drawing one included — this is the object that knows there is a remote service behind it,
and unguarded, a matrix would be uploaded once per device.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from src.tracking.keys import SEGMENT, MetricKey
from src.tracking.registry import tracker_registry

if TYPE_CHECKING:
    from argparse import Namespace

    from clearml import Task
    from clearml.logger import Logger as Backend

    from src.core import Matrix

log = logging.getLogger(__name__)

DECIMALS = 3
"""Matrix cells are read rather than computed with: 0.333 reads, 0.3333333 does not."""

SERIES = "value"
"""The line a value that names none of its own is drawn as — `epoch` is a graph with one series."""


@tracker_registry.register("clearml")
class ClearMLTracker(Logger):
    """One ClearML task carrying a run's numbers and the readings that mean a picture.

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
        from clearml import Task

        self._task: Task = Task.init(
            project_name=project_name,
            task_name=task_name,
            tags=_worth_showing(tags or ()),
            reuse_last_task_id=reuse_last_task_id,
            **options,
        )
        self._backend: Backend = self._task.get_logger()

    @property
    def name(self) -> str:
        return str(self._task.name)

    @property
    def version(self) -> str:
        return str(self._task.id)

    @rank_zero_only
    def log_hyperparams(self, params: Mapping[str, Any] | Namespace, *args: Any, **kwargs: Any) -> None:
        self._task.connect(dict(params) if isinstance(params, Mapping) else vars(params))

    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        for key, value in metrics.items():
            title, series = _drawn_as(key)
            self._backend.report_scalar(title=title, series=series, value=float(value), iteration=step or 0)

    @rank_zero_only
    def log_matrix(self, title: str, matrix: Matrix, iteration: int) -> None:
        labels = list(matrix.labels) if matrix.labels is not None else None
        self._backend.report_confusion_matrix(
            title=title,
            series=SERIES,
            matrix=matrix.value.detach().cpu().float().round(decimals=DECIMALS).numpy(),
            iteration=iteration,
            xlabels=labels,
            ylabels=labels,
            xaxis=matrix.xaxis,
            yaxis=matrix.yaxis,
        )

    @rank_zero_only
    def finalize(self, status: str) -> None:
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
        return key, SERIES
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
