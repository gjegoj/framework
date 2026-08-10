"""Rich progress bar with a live per-metric summary table."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from lightning.pytorch.callbacks.progress.rich_progress import (
    _IS_INTERACTIVE,
    CustomProgress,
    RichProgressBar,
)
from rich.console import Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from src.console import HEADER_STYLE
from src.core import log_keys
from src.core.ports import DeclaresMetricDirections
from src.core.taxonomy import Stage

if TYPE_CHECKING:
    from collections.abc import Mapping

type _MetricMode = Literal["min", "max"]

# Test is a single post-fit pass: it gets a value column but no running "best" —
# that only means something for stages repeated across epochs.
_BEST_STAGES: frozenset[str] = frozenset({Stage.TRAIN, Stage.VAL})

_REFRESH_RATE = 4  # Hz — how often the Live display redraws


def row_key(logged: str) -> str | None:
    """Normalize a logged key to its table row, or ``None`` for keys the table skips.

    Scalars pass as they are; a vector metric's ``mean`` collapses to the
    plain key (the row shows the aggregate); per-class leaves and stage-less
    keys (``epoch``) are noise at table altitude.
    """
    stage, separator, rest = logged.partition(log_keys.SEPARATOR)
    if not separator or stage not in log_keys.STAGES:
        return None
    segments = rest.split(log_keys.SEPARATOR)
    if len(segments) < 3:
        return logged
    if len(segments) == 3 and segments[-1] == log_keys.MEAN:
        return logged.rsplit(log_keys.SEPARATOR, 1)[0]
    return None


class MetricHistory:
    """Current values, per-step deltas, and direction-aware bests of logged scalars.

    Plain Python on purpose — the tracking is testable without Lightning or
    rich, and the bar stays display glue.

    Directions are never guessed from a name. They are declared by the module,
    and what it does not declare is a loss: at table altitude this framework logs
    its metrics, which are declared, and its losses, which are not — a total, its
    parts, and whatever a decorated model adds to them. A metric declared
    *directionless* still tracks no best, which is the half of that rule doing
    real work.
    """

    def __init__(self, directions: Mapping[str, bool | None] | None = None) -> None:
        self._directions = dict(directions or {})
        self.current: dict[str, float] = {}
        self.step_deltas: dict[str, float] = {}
        self.best: dict[str, float] = {}
        self.best_deltas: dict[str, float] = {}

    def declare(self, directions: Mapping[str, bool | None]) -> None:
        """Adopt declared directions without forgetting what has already been observed."""
        self._directions.update(directions)

    def direction(self, key: str) -> _MetricMode | None:
        """The declared optimization mode, or ``min`` for the losses nobody declares.

        Undeclared means a loss, and a loss is minimized by construction — the
        optimizer descends the weighted sum of exactly those parts. Reading it
        this way rather than from a list the module keeps costs nothing and is
        blind to where a part came from, so a decorated model's own term is
        signed like any other.

        The premise is that metrics and losses are all this framework logs at
        table altitude; ``test_nothing_but_a_loss_arrives_undeclared`` names the
        keys that rely on it, so a fourth kind fails there rather than appearing
        with a best it never earned.
        """
        if key not in self._directions:
            return "min"
        flag = self._directions[key]
        return None if flag is None else ("max" if flag else "min")

    def observe(self, key: str, value: float) -> None:
        """Record one observation: step delta, then direction-aware best tracking."""
        previous = self.current.get(key)
        if previous is not None and value != previous:
            self.step_deltas[key] = value - previous
        self.current[key] = value

        direction = self.direction(key)
        if direction is None:
            return
        best = self.best.get(key)
        if best is None:
            self.best[key] = value
            return
        improved = (value < best) if direction == "min" else (value > best)
        if improved:
            self.best_deltas[key] = value - best
            self.best[key] = value


class MetricsProgressBar(RichProgressBar):
    """``RichProgressBar`` with a live metrics table rendered below the bar.

    Each row is one series (``label/f1``, ``loss``) with its current Train/Val/Test
    values, the best Train/Val values observed, and colour-coded deltas beside them.
    Which direction counts as an improvement is the metric's own declared
    ``higher_is_better``, asked from the module through the ``DeclaresMetricDirections``
    port and never guessed from a name; what the module does not declare is a loss, and
    a loss is minimized.

    Parameters:
        metric_filters (list[str] | None): Substrings narrowing the table; a
            key is shown when its name contains any entry. ``None`` shows
            every table-shaped key.
        **kwargs: Forwarded verbatim to ``RichProgressBar``.
    """

    def __init__(self, metric_filters: list[str] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._metric_filters = metric_filters
        self._live: Live | None = None  # the progress + table group; created in _init_progress
        self._history = MetricHistory()

    def setup(self, trainer: Any, pl_module: Any, stage: str) -> None:
        """Learn which metrics the module declares, keeping what earlier stages recorded.

        Told, never replaced: ``setup`` runs again for ``test``, and starting over
        there would empty the Train and Val columns exactly when the Test column
        arrives — and those columns exist to be read against it.
        """
        super().setup(trainer, pl_module, stage)
        if isinstance(pl_module, DeclaresMetricDirections):
            self._history.declare(pl_module.metric_directions())

    def _init_progress(self, trainer: Any) -> None:
        """Wrap the progress bar and the metrics table in one shared Live group."""
        if not self.is_enabled or (self.progress is not None and not self._progress_stopped):
            return
        self._reset_progress_bar_ids()
        self.progress = CustomProgress(
            *self.configure_columns(trainer),
            auto_refresh=False,
            disable=self.is_disabled,
            console=self._console,
        )
        self._live = Live(
            Group(self.progress, self._build_table()),
            refresh_per_second=_REFRESH_RATE,
            console=self._console,
        )
        self._live.start()
        self._progress_stopped = False

    def refresh(self, hard: bool = False) -> None:
        """Refresh the bar and re-render the table from the latest logged values.

        Mirrors ``RichProgressBar.refresh``: a hard (or interactive) refresh
        redraws fully, a soft one avoids flicker.

        The table reads ``callback_metrics``, not the bar's own ``get_metrics``.
        The latter serves ``progress_bar_metrics`` — only what was logged with
        ``prog_bar=True``, which here is the training loss and nothing else, so a
        table fed from it has five columns it can never fill. Measured: at the
        end of a validation epoch and after ``test``, ``progress_bar_metrics`` is
        empty while ``callback_metrics`` holds every key the run logged, named
        exactly as this table's rows are.
        """
        if not self.progress:
            return
        if hard or _IS_INTERACTIVE:
            self.progress.refresh()
        else:
            self.progress.soft_refresh()
        if self._live is None:
            return

        for name, raw in self.trainer.callback_metrics.items():
            key = row_key(name)
            if key is None:
                continue
            if self._metric_filters is not None and not any(token in name for token in self._metric_filters):
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            self._history.observe(key, value)
        self._live.update(Group(self.progress, self._build_table()))

    def _build_table(self) -> Table:
        """Assemble the Metric x Train/Best/Val/Best/Test table from everything seen so far.

        Read from the history rather than from the values of this one refresh:
        Lightning empties ``callback_metrics`` between ``fit`` and ``test``, so a
        table built from the latest batch of keys alone would blank the Train and
        Val columns at the moment the Test column arrives.
        """
        table = Table(show_header=True, header_style=HEADER_STYLE)
        table.add_column("Metric")
        table.add_column("Train", justify="right")
        table.add_column("Best (train)", justify="right")
        table.add_column("Val", justify="right")
        table.add_column("Best (val)", justify="right")
        table.add_column("Test", justify="right")

        rows: dict[str, dict[str, Text]] = {}
        for key, value in self._history.current.items():
            stage, _, series = key.partition(log_keys.SEPARATOR)
            rows.setdefault(series, {})[stage] = self._cell(key, value, self._history.step_deltas)
        for key, best in self._history.best.items():
            stage, _, series = key.partition(log_keys.SEPARATOR)
            if stage not in _BEST_STAGES:
                continue
            rows.setdefault(series, {})[f"{stage}_best"] = self._cell(key, best, self._history.best_deltas)

        for series in sorted(rows):
            row = rows[series]
            table.add_row(
                Text(series),
                row.get(Stage.TRAIN),
                row.get(f"{Stage.TRAIN}_best"),
                row.get(Stage.VAL),
                row.get(f"{Stage.VAL}_best"),
                row.get(Stage.TEST),
            )
        return table

    def _cell(self, key: str, value: float | None, deltas: Mapping[str, float]) -> Text:
        """One table cell: the value, then a colour-coded delta arrow when it moved."""
        if value is None:
            return Text("-")
        cell = Text(f"{value:.4f}")
        direction = self._history.direction(key)
        delta = deltas.get(key)
        if direction is None or delta is None or delta == 0.0:
            return cell
        improved = (delta < 0) if direction == "min" else (delta > 0)
        arrow = "▼" if delta < 0 else "▲"
        cell.append(f" {arrow}{abs(delta):.4f}", style="green" if improved else "red")
        return cell

    def on_test_end(self, trainer: Any, pl_module: Any) -> None:
        """Render once more after test metrics finalize.

        ``RichProgressBar`` refreshes on train/val epoch ends but never after
        test, so the just-computed test column would stay empty without this.
        """
        self.refresh(hard=True)
        super().on_test_end(trainer, pl_module)

    def teardown(self, trainer: Any, pl_module: Any, stage: str) -> None:
        """Stop the Live group cleanly and leave a trailing blank line.

        The base class stops only its own ``progress``; our group would be
        torn down at interpreter exit, leaving the cursor glued to the table's
        bottom border so the next output starts on the same line.
        """
        if self._live is not None:
            self._live.stop()
            if self._console is not None:
                self._console.print()
            self._live = None
        super().teardown(trainer, pl_module, stage)
