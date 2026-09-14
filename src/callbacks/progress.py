"""Lightning's progress bar, with the run's own numbers in the same live region.

Three things kept apart on purpose: what has been seen (:class:`MetricHistory`), how it is drawn
(:func:`table`) and where it is drawn (:class:`MetricsProgressBar`). The first two are plain Python
and need neither a trainer nor a terminal, which is why the rules a reader depends on — which way a
measurement is better, what counts as a move — are testable without rendering anything at all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, override

from lightning.pytorch.callbacks import RichProgressBar
from rich.console import Group
from rich.table import Table
from rich.text import Text

from src.callbacks.registry import callback_registry
from src.core import Stage
from src.tracking import MetricKey
from src.training import DeclaresMetricDirections

if TYPE_CHECKING:
    import lightning as L

COLUMNS: tuple[tuple[Stage, bool], ...] = (
    (Stage.TRAIN, False),
    (Stage.TRAIN, True),
    (Stage.VAL, False),
    (Stage.VAL, True),
    (Stage.TEST, False),
)
"""Each column as the stage it reads and whether it holds that stage's best so far.

Test carries no best because it is one pass after the fit: the best of a single reading is that
reading, printed twice. Saying it here rather than as a rule elsewhere means the table's shape and
the claim about it are the same line.
"""

CELL_DECIMALS = 4
"""Digits a reading is shown with: enough to see an epoch move, few enough to read a column."""


class MetricHistory:
    """What each measurement last said, how far it moved, and the best of it so far.

    Directions are declared by whoever computes the measurements and never guessed from a name —
    in this framework a name says nothing about a direction, and guessing it shows the wrong best.
    """

    def __init__(self, directions: Mapping[str, bool | None] | None = None) -> None:
        self._directions = dict(directions or {})
        self.current: dict[MetricKey, float] = {}
        self.moves: dict[MetricKey, float] = {}
        self.best: dict[MetricKey, float] = {}
        self.best_moves: dict[MetricKey, float] = {}

    def declare(self, directions: Mapping[str, bool | None]) -> None:
        """Take directions in without forgetting what has already been seen.

        Told rather than replaced: a module declares again when the test stage is set up, and
        starting over there would blank the very columns the test column is read against.
        """
        self._directions.update(directions)

    def better_higher(self, key: MetricKey) -> bool | None:
        """Whether a larger reading is the better one, or ``None`` where there is no better at all.

        What nobody declares is a loss: the objective and its terms are logged by the loop itself and
        are measured by no metric, so there is nothing else an undeclared number here could be.
        """
        return self._directions.get(key.series, False)

    def observe(self, key: MetricKey, value: float) -> None:
        """Record one reading: where it moved, and then the best of it where a best means something."""
        previous = self.current.get(key)
        if previous is not None and value != previous:
            self.moves[key] = value - previous
        self.current[key] = value

        better_higher = self.better_higher(key)
        if better_higher is None:
            return
        best = self.best.get(key)
        if best is None:
            self.best[key] = value
        elif (value > best) if better_higher else (value < best):
            self.best_moves[key] = value - best
            self.best[key] = value


def table(history: MetricHistory) -> Table:
    """Everything seen so far as one row per measurement, read across the stages that produced it.

    Built from the history rather than from the latest reading of each key: Lightning empties its
    metrics between the fit and the test, so a table assembled from what is currently logged would
    blank the train and val columns at the exact moment the test column arrives to be compared.
    """
    built = Table(show_header=True)
    built.add_column("Metric")
    for stage, best in COLUMNS:
        built.add_column(f"Best ({stage})" if best else stage.capitalize(), justify="right")

    rows: dict[str, dict[tuple[Stage, bool], Text]] = {}
    for key, value in history.current.items():
        rows.setdefault(key.series, {})[key.stage, False] = _cell(value, history.moves.get(key), history, key)
    for key, value in history.best.items():
        rows.setdefault(key.series, {})[key.stage, True] = _cell(value, history.best_moves.get(key), history, key)

    for series in sorted(rows):
        built.add_row(Text(series), *(rows[series].get(column, _MISSING) for column in COLUMNS))
    return built


_MISSING = Text("—")
"""A stage that has not measured this yet — which is most of them, most of the time."""


def _cell(value: float, moved: float | None, history: MetricHistory, key: MetricKey) -> Text:
    """One reading, and where it moved when it moved; coloured by whether the move was an improvement."""
    cell = Text(f"{value:.{CELL_DECIMALS}f}")
    better_higher = history.better_higher(key)
    if moved is None or moved == 0.0 or better_higher is None:
        return cell
    improved = moved > 0 if better_higher else moved < 0
    arrow = "▲" if moved > 0 else "▼"
    cell.append(f" {arrow}{abs(moved):.{CELL_DECIMALS}f}", style="green" if improved else "red")
    return cell


@callback_registry.register("progress")
class MetricsProgressBar(RichProgressBar):
    """The bar Lightning draws, with :func:`table` drawn under it and refreshed with it.

    Parameters:
        metric_filters: Show only rows whose name contains one of these; ``None`` shows every one.
            A run measuring three tasks writes a long table, and a reader usually watches two rows.
        **options: Forwarded verbatim to ``RichProgressBar``, so every knob of it stays reachable.
    """

    def __init__(self, metric_filters: Sequence[str] | None = None, **options: Any) -> None:
        super().__init__(**options)
        self._filters = None if metric_filters is None else tuple(metric_filters)
        self._history = MetricHistory()

    @property
    def history(self) -> MetricHistory:
        """The numbers this has accumulated, for whoever wants them without a terminal to read them off."""
        return self._history

    @override
    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        if isinstance(pl_module, DeclaresMetricDirections):
            self._history.declare(pl_module.metric_directions())

    @override
    def get_metrics(self, trainer: L.Trainer, pl_module: L.LightningModule) -> dict[str, Any]:
        """Nothing beside the bar: everything a run measures is already under it, and said better.

        Lightning's own answer here is the logger's version plus whatever was logged with
        ``prog_bar=True`` — beside this bar that is a second, shorter rendering of one row of the
        table below it, without the stage it belongs to, without its best and without which way it
        moved. ``v_num`` names the logger's version, which the run directory is already named after.

        The module's ``prog_bar`` request is not withdrawn by this and is not dead: a run declaring no
        progress callback gets Lightning's own bar, which honours it and draws no table.
        """
        return {}

    @override
    def _init_progress(self, trainer: L.Trainer) -> None:
        """Let Lightning build its bar, then give the live display a renderable that has the table too.

        *How* the bar is built stays entirely Lightning's — its columns, its refresh thread, the
        compatibility shims it carries for rich — because a copy of that here is a copy that goes
        stale: the version of this in the framework it grew out of still built the bar the way an
        older Lightning did, and quietly lost the metrics column that Lightning has added since.
        What is added is the one line below, on rich's own terms: a live display takes its renderable
        from a callable, and this hands it one that draws the bar with the table underneath.
        """
        super()._init_progress(trainer)
        if self.progress is not None:
            progress = self.progress
            progress.live._get_renderable = lambda: Group(progress.get_renderable(), table(self._history))

    @override
    def refresh(self, hard: bool = False) -> None:
        """Take in whatever has been logged since the last redraw, then let the base class redraw.

        Read here rather than in an epoch hook because a callback's hooks run *before* the module's:
        at the end of an epoch the numbers about to be shown are not in ``callback_metrics`` yet, and
        a refresh is the one moment that is always after them.
        """
        if self.progress is not None:
            self._observe()
        super().refresh(hard)

    @override
    def teardown(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        """Take the numbers in once more, before the display stops.

        A module reports its epoch *after* every callback has seen the end of one, so what a stage
        finished with is not logged yet when the last hook of that stage runs — and a fit ends on such
        an epoch, as does a test. This is the last hook of all, and without it the table a reader is
        left looking at is missing the very column that was just filled.
        """
        self.refresh(hard=True)
        super().teardown(trainer, pl_module, stage)

    def _observe(self) -> None:
        """Every logged key that is a measurement, at the reading a table shows it by.

        Read as a float without a guard, because a value that is not one never reaches here —
        measured: ``self.log`` refuses anything but a single-element tensor.
        """
        for logged, value in self.trainer.callback_metrics.items():
            key = MetricKey.headline(logged)
            if key is None or not self._shows(key):
                continue
            self._history.observe(key, float(value))

    def _shows(self, key: MetricKey) -> bool:
        return self._filters is None or any(token in key.series for token in self._filters)
