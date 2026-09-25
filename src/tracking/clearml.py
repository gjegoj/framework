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

from src.core import BYTES_PER_GIB
from src.tracking.keys import SEGMENT, MetricKey
from src.tracking.registry import tracker_registry

if TYPE_CHECKING:
    from argparse import Namespace
    from pathlib import Path

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
        media_uri: Where a page is uploaded to, for a run whose service cannot take one. A page is a
            debug sample, and a debug sample goes to ``api.files_server`` and nowhere else — read in
            clearml 2.1.10: ``output_uri`` is documented for models and artifacts, and no key of
            ``clearml.conf`` reaches it. What does is the logger's own destination, which is why this
            is the one parameter here the service is never handed. An artifact goes here too when the task
            has no ``output_uri`` — read in 2.1.10: ``upload_artifact`` uses ``task.output_uri`` or the
            logger's default destination, which this sets; ``task.output_uri`` is set by a declared one, by
            ``sdk.development.default_output_uri`` in ``clearml.conf``, or by the project's default.
        keep_suffixes: Which of the model's files this run keeps here, by suffix — without the dot and in lower
            case, as ``export`` spells it; a checkpoint goes by its own file's, ``ckpt`` for one Lightning wrote.
            ``None`` keeps every one, an empty list none.
        keep_max_gib: The size of a whole format above which it stays on disk, named in a warning; ``None`` for
            no limit. One by default: a student's checkpoint is tens of megabytes, a ViT-L teacher's with its
            optimizer and average is gigabytes, and an upload that size at the end of every run should be asked
            for rather than waited through.
        **options: Forwarded to ``Task.init`` verbatim, so every upstream knob stays reachable.
    """

    def __init__(
        self,
        project_name: str | None = None,
        task_name: str | None = None,
        tags: Sequence[str] | None = None,
        reuse_last_task_id: bool = False,
        media_uri: str | None = None,
        *,
        keep_suffixes: Sequence[str] | None = None,
        keep_max_gib: float | None = 1.0,
        **options: Any,
    ) -> None:
        super().__init__()
        if isinstance(keep_suffixes, str) or any(one != one.lstrip(".").lower() for one in keep_suffixes or ()):
            raise ValueError(
                "`tracker.keep_suffixes` lists suffixes without their dot and in lower case, as `export` spells "
                f"them, e.g. [onnx, ckpt]; got {keep_suffixes!r}."
            )
        if keep_max_gib is not None and keep_max_gib <= 0:
            raise ValueError(
                f"`tracker.keep_max_gib` is the size above which a format stays on disk, so it is positive; got "
                f"{keep_max_gib}. Write `null` for no limit."
            )
        frameworks = options.pop("auto_connect_frameworks", {"pytorch": False})
        if frameworks is True or (isinstance(frameworks, Mapping) and frameworks.get("pytorch")):
            raise ValueError(
                f"`tracker.auto_connect_frameworks: {frameworks!r}` has ClearML capture the model's weights itself — "
                "every epoch's checkpoint and the one read back, never what `export` wrote — and which of them reach "
                "the service is `tracker.keep_suffixes`. Leave `pytorch` out of the mapping, or the key out altogether."
            )
        self._declared: dict[str, Any] = {
            "project_name": project_name,
            "task_name": task_name,
            "tags": _worth_showing(tags or ()),
            "reuse_last_task_id": reuse_last_task_id,
            **options,
            # Measured (spec 2026-09-25): left on, ClearML's PyTorch hook files every epoch's checkpoint as an
            # output model and the one read back as an input model. A framework a mapping does not name it
            # captures, and a mapping that is empty — like `False` or `None` — it reads as nothing at all, so
            # `pytorch` is added to a mapping and never to a declaration that already turns everything off.
            "auto_connect_frameworks": {**frameworks, "pytorch": False} if frameworks else frameworks,
        }
        self._media_uri = media_uri
        self._keep_suffixes = None if keep_suffixes is None else frozenset(keep_suffixes)
        self._keep_max_gib = keep_max_gib
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
            if self._media_uri is not None:
                # Once, on the logger this run keeps: it is made here and held, so every page that
                # follows goes the same way. Credentials are `clearml.conf`'s, and a destination it
                # cannot write to is refused on the spot — which is when the run is created, before
                # the first epoch, rather than at whichever epoch first draws.
                self._task.get_logger().set_default_upload_destination(self._media_uri)
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
    def log_record(self, name: str, record: Mapping[str, object]) -> None:
        """The ``KeepsRecord`` port: a record this service keeps as an artifact of the task.

        An artifact and not media: this is read by whoever deploys the model, often from a script, and
        the service stores a mapping as something they can fetch back as one.
        """
        self.experiment.upload_artifact(name, dict(record))

    @rank_zero_only
    def log_file(self, path: Path, travels_with: Sequence[Path] = ()) -> None:
        """The ``KeepsFiles`` port: a format and what travels with it, as artifacts of the task.

        An artifact rather than an ``OutputModel``: read in 2.1.10, a model with no ``output_uri`` registers the
        path on the machine that trained and uploads nothing, while an artifact goes wherever the record already
        goes — the task's ``output_uri`` (declared, or from ``clearml.conf`` or the project), else ``media_uri``,
        else the file server.

        Each file is waited for, because this is the last thing a run does, after Lightning's ``finalize``, and a
        line saying a file is on the service has to mean it is. Queuing them all and waiting once would save the
        upload time of every file but the largest, over one shared link, at the price of a failure only
        ClearML's own log would see. The first file the service does not take stops the format — what follows
        would only add to half a model there — and the warning says how much of the format did arrive.
        """
        whole = (path, *travels_with)
        if self._keep_suffixes is not None and path.suffix.removeprefix(".") not in self._keep_suffixes:
            log.info(
                "%s stays on disk: `tracker.keep_suffixes` keeps %s.",
                path.name,
                ", ".join(sorted(self._keep_suffixes)) or "nothing",
            )
            return
        gib = sum(one.stat().st_size for one in whole) / BYTES_PER_GIB
        if self._keep_max_gib is not None and gib > self._keep_max_gib:
            log.warning(
                "%s is %.2f GiB with what travels with it, above `tracker.keep_max_gib: %s`; it stays in %s. Raise "
                "the limit, or write `keep_max_gib: null`, to keep it on the service.",
                path.name,
                gib,
                self._keep_max_gib,
                path.parent,
            )
            return
        for sent, one in enumerate(whole):
            try:
                taken = self.experiment.upload_artifact(one.name, artifact_object=one, wait_on_upload=True)
            except Exception as error:
                taken, why = False, str(error)
            else:
                why = "the service declined it"
            if not taken:
                log.warning(
                    "ClearML could not keep %s (%s); %d of the %d files of %s reached it, and all of them are still "
                    "in %s.",
                    one.name,
                    why,
                    sent,
                    len(whole),
                    path.name,
                    path.parent,
                )
                return
            log.info("%s is kept on the service.", one.name)

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
