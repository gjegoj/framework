"""Rich terminal table builders for dataset distribution reporting.

Moved verbatim from ``src/callbacks/dataset_stats.py`` — terminal output is
byte-for-byte identical to the original. The functions are now public so
``DistributionRenderer`` implementations can call them directly.
"""

from __future__ import annotations

from rich.table import Table

from src.core.enums import Stage
from src.data.statistics import CategoricalDistribution, ContinuousDistribution

__all__ = ["new_table", "categorical_table", "continuous_table", "CONTINUOUS_ROWS"]


# Ordered (display label, ContinuousDistribution attribute) pairs.
CONTINUOUS_ROWS: tuple[tuple[str, str], ...] = (
    ("count", "count"),
    ("mean", "mean"),
    ("std", "std"),
    ("min", "minimum"),
    ("q25", "q25"),
    ("median", "median"),
    ("q75", "q75"),
    ("max", "maximum"),
)


def new_table(title: str, first_column: str, stages: list[Stage]) -> Table:
    """A titled table with a label column and one right-aligned column per stage.

    Parameters:
        title (str): Table title, left-justified above the header row.
        first_column (str): Header text for the leftmost (label) column.
        stages (list[Stage]): Ordered stages that become additional columns.

    Returns:
        Table: An empty ``rich.table.Table`` ready for data rows.
    """
    table = Table(title=title, title_justify="left", header_style="bold magenta")  # match the progress bar
    table.add_column(first_column)
    for stage in stages:
        table.add_column(stage.value.capitalize(), justify="right")
    return table


def categorical_table(
    task_name: str,
    per_stage: dict[Stage, CategoricalDistribution],
    stages: list[Stage],
) -> Table:
    """Build a class-distribution table for a categorical task.

    Parameters:
        task_name (str): Task name shown in the table title.
        per_stage (dict[Stage, CategoricalDistribution]): Distribution per stage.
        stages (list[Stage]): Ordered stages to include as columns.

    Returns:
        Table: A rich table with one row per class plus a bold totals row.
    """
    table = new_table(f"{task_name}  (class distribution)", "Class", stages)
    class_names = list(per_stage[stages[0]].counts)  # class-index order, shared across stages
    totals = {stage: per_stage[stage].total for stage in stages}  # once per stage, not per cell
    for class_name in class_names:
        cells = [class_name]
        for stage in stages:
            count = per_stage[stage].counts[class_name]
            percent = (100 * count / totals[stage]) if totals[stage] else 0.0
            cells.append(f"{count} ({percent:.1f}%)")
        table.add_row(*cells)
    table.add_row("Total", *(str(totals[stage]) for stage in stages), style="bold")
    return table


def continuous_table(
    task_name: str,
    per_stage: dict[Stage, ContinuousDistribution],
    stages: list[Stage],
) -> Table:
    """Build a numeric-summary table for a continuous task.

    Parameters:
        task_name (str): Task name shown in the table title.
        per_stage (dict[Stage, ContinuousDistribution]): Distribution per stage.
        stages (list[Stage]): Ordered stages to include as columns.

    Returns:
        Table: A rich table with one row per summary statistic.
    """
    table = new_table(f"{task_name}  (numeric)", "Statistic", stages)
    for label, attribute in CONTINUOUS_ROWS:
        cells = [label]
        for stage in stages:
            value = getattr(per_stage[stage], attribute)
            cells.append(str(int(value)) if attribute == "count" else f"{value:.3f}")
        table.add_row(*cells)
    return table
