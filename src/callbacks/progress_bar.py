"""Rich progress bar with a live per-metric summary table."""

from __future__ import annotations

from typing import Any, Literal

from lightning.pytorch.callbacks.progress.rich_progress import (
    _IS_INTERACTIVE,
    CustomProgress,
    RichProgressBar,
)
from rich.console import Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from src.core.ports import MetricDirectionProvider

_MetricMode = Literal["min", "max"]

_TRAINING_STAGES: frozenset[str] = frozenset({"train", "val"})

_REFRESH_RATE = 4  # Hz — how often the Live display redraws


def _mode_from_flag(higher_is_better: bool | None) -> _MetricMode | None:
    """Translate a metric's ``higher_is_better`` flag into an optimization mode."""
    if higher_is_better is None:
        return None
    return "max" if higher_is_better else "min"


def _split_stage(metric_name: str) -> tuple[str, str | None]:
    """Separate a ``task/metric/stage`` key into ``(task/metric, stage)``."""
    parts = metric_name.split("/")
    for stage in _TRAINING_STAGES:
        if stage in parts:
            stage_index = parts.index(stage)
            base_name = "/".join(parts[:stage_index] + parts[stage_index + 1 :])
            return base_name, stage
    return metric_name, None


class MetricsProgressBar(RichProgressBar):
    """RichProgressBar with a live metrics table rendered below the progress bar.

    Each row shows a metric's current Train/Val values alongside its best
    observed values. Color-coded directional deltas (▲▼) indicate whether the
    latest change is an improvement. Improvement direction is the metric's own
    declared ``higher_is_better`` flag, bound once at ``setup`` from the module's
    tasks; losses (which are not metrics) default to "lower is better".

    Parameters:
        loss_key: Namespace prefix that identifies loss keys (the first path
            segment, e.g. ``loss/train/total``). Losses are always displayed and
            treated as "lower is better".
        metric_filters: Additional substrings; a metric is shown when its name
            contains any entry. Pass ``None`` to auto-show all three-part keys
            matching ``<task>/<metric>/<stage>``.
        **kwargs: Forwarded verbatim to ``RichProgressBar``.
    """

    def __init__(
        self,
        loss_key: str = "loss",
        metric_filters: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._loss_key = loss_key
        self._metric_filters = metric_filters

        self._direction_by_key: dict[str, _MetricMode | None] = {}
        self._current_values: dict[str, float] = {}
        self._step_deltas: dict[str, float] = {}
        self._best_values: dict[str, float] = {}
        self._best_deltas: dict[str, float] = {}

    # ------------------------------------------------------------------ helpers

    def _bind_directions(self, pl_module: Any) -> None:
        """Record each metric's declared direction from the module.

        Asks a ``MetricDirectionProvider`` for its ``higher_is_better`` flags,
        already keyed by the ``task/metric/stage`` string the module logs, so no
        direction is guessed from the metric's name. A module that does not
        provide directions simply contributes nothing.
        """
        if not isinstance(pl_module, MetricDirectionProvider):
            return
        self._direction_by_key = {key: _mode_from_flag(flag) for key, flag in pl_module.metric_directions().items()}

    def _direction_for(self, metric_name: str) -> _MetricMode | None:
        if metric_name in self._direction_by_key:
            return self._direction_by_key[metric_name]
        if self._loss_key and metric_name.split("/", 1)[0] == self._loss_key:
            return "min"
        return None

    def _is_displayed(self, metric_name: str) -> bool:
        if self._loss_key and metric_name.split("/", 1)[0] == self._loss_key:
            return True
        parts = metric_name.split("/")
        if self._metric_filters is None:
            return len(parts) == 3 and parts[-1] in _TRAINING_STAGES
        return any(filter_substring in metric_name for filter_substring in self._metric_filters)

    def _track(self, metric_name: str, value: float) -> None:
        """Record a new observation: update step delta and best-value tracking."""
        previous = self._current_values.get(metric_name)
        if previous is not None and value != previous:
            self._step_deltas[metric_name] = value - previous
        self._current_values[metric_name] = value

        direction = self._direction_for(metric_name)
        if direction is None:
            return

        best = self._best_values.get(metric_name)
        if best is None:
            self._best_values[metric_name] = value
            return

        improved = (value < best) if direction == "min" else (value > best)
        if improved:
            self._best_deltas[metric_name] = value - best
            self._best_values[metric_name] = value

    def _format_cell(
        self,
        metric_name: str,
        value: float | None,
        delta_map: dict[str, float],
    ) -> Text:
        """Render a table cell: value followed by a color-coded delta arrow."""
        if value is None:
            return Text("-")

        direction = self._direction_for(metric_name)
        delta = delta_map.get(metric_name)
        cell = Text(f"{value:.4f}")

        if direction is None or delta is None or delta == 0.0:
            return cell

        improved = (delta < 0) if direction == "min" else (delta > 0)
        arrow = "▼" if delta < 0 else "▲"
        cell.append(f" {arrow}{abs(delta):.4f}", style="green" if improved else "red")
        return cell

    def _build_table(self, displayed_metrics: dict[str, float]) -> Table:
        """Assemble the Train/Val x Current/Best summary table."""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric")
        table.add_column("Train", justify="right")
        table.add_column("Best (train)", justify="right")
        table.add_column("Val", justify="right")
        table.add_column("Best (val)", justify="right")

        rows: dict[str, dict[str, Text]] = {}

        for name, value in displayed_metrics.items():
            base, stage = _split_stage(name)
            if stage not in _TRAINING_STAGES:
                continue
            rows.setdefault(base, {})[stage] = self._format_cell(name, value, self._step_deltas)

        for name, best_value in self._best_values.items():
            base, stage = _split_stage(name)
            if stage not in _TRAINING_STAGES:
                continue
            rows.setdefault(base, {})[f"{stage}_best"] = self._format_cell(name, best_value, self._best_deltas)

        for base_name in sorted(rows):
            row = rows[base_name]
            table.add_row(
                Text(base_name),
                row.get("train"),
                row.get("train_best"),
                row.get("val"),
                row.get("val_best"),
            )

        return table

    # ------------------------------------------------------- lightning overrides

    def setup(self, trainer: Any, pl_module: Any, stage: str) -> None:
        """Bind metric directions from the module once training/eval is set up."""
        super().setup(trainer, pl_module, stage)
        self._bind_directions(pl_module)

    def _init_progress(self, trainer: Any) -> None:
        """Override to wrap progress bar and metrics table in a shared Live group."""
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
            Group(self.progress, self._build_table({})),
            refresh_per_second=_REFRESH_RATE,
            console=self._console,
        )
        self._live.start()
        self._progress_stopped = False

    def refresh(self, hard: bool = False) -> None:
        """Refresh the progress bar and re-render the metrics table.

        Mirrors ``RichProgressBar.refresh``: a hard (or interactive) refresh
        redraws fully, otherwise a soft refresh avoids flicker. The metrics
        table is rebuilt from the latest logged values on every call.
        """
        if not self.progress:
            return
        if hard or _IS_INTERACTIVE:
            self.progress.refresh()
        else:
            self.progress.soft_refresh()

        if self._live is None:
            return

        raw_metrics = self.get_metrics(self.trainer, self.trainer.lightning_module)
        raw_metrics.pop("v_num", None)

        displayed: dict[str, float] = {}
        for name, raw_value in raw_metrics.items():
            if not self._is_displayed(name) or isinstance(raw_value, dict):
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            self._track(name, value)
            displayed[name] = value

        self._live.update(Group(self.progress, self._build_table(displayed)))
