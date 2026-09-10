"""One table into named splits: at random, keeping class shares, or keeping whole groups together."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import IterativeStratification

from src.core import validate_name
from src.data.base import Table
from src.data.encoders.label import labels_in

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Split:
    """How to divide: fractions per split name (summing to one), and at most one balancing rule.

    ``seed`` is separate from the experiment's on purpose: runs at different seeds must share
    one test set, or their numbers are not comparable.
    """

    fractions: Mapping[str, float]
    seed: int = 42
    stratify_by: str | None = None
    stratify_bins: int = 10
    stratify_separator: str = ","
    group_by: str | None = None

    def __post_init__(self) -> None:
        if not self.fractions:
            raise ValueError("A split needs at least one named fraction.")
        for name in self.fractions:
            validate_name(name, kind="Split")
        if any(fraction < 0 for fraction in self.fractions.values()):
            raise ValueError("Split fractions are nonnegative.")
        if not math.isclose(sum(self.fractions.values()), 1.0, abs_tol=1e-6):
            raise ValueError(f"Split fractions must sum to 1, got {sum(self.fractions.values())}.")
        if self.stratify_by is not None and self.group_by is not None:
            raise ValueError("stratify_by and group_by cannot be combined: a group moves whole, a stratum is spread.")
        if self.stratify_bins < 2:
            raise ValueError(f"stratify_bins needs at least 2, got {self.stratify_bins}.")


def split_table(table: Table, split: Split) -> dict[str, Table]:
    rows = table.reset_index(drop=True)
    if split.group_by is not None:
        return _by_groups(rows, split)
    if split.stratify_by is not None:
        return _stratified(rows, split)
    return _at_random(rows, split)


def _at_random(rows: Table, split: Split) -> dict[str, Table]:
    shuffled = rows.sample(frac=1, random_state=split.seed).reset_index(drop=True)
    parts: dict[str, Table] = {}
    names = list(split.fractions)
    start = 0
    for position, name in enumerate(names):
        end = len(shuffled) if position == len(names) - 1 else start + int(len(shuffled) * split.fractions[name])
        parts[name] = shuffled.iloc[start:end].reset_index(drop=True)
        start = end
    return _refusing_empty(parts, f"{len(rows)} rows do not stretch across the requested fractions")


def _stratified(rows: Table, split: Split) -> dict[str, Table]:
    column = _column(rows, split.stratify_by, purpose="stratify")
    indicators = _label_indicators(rows[column], split.stratify_separator)
    if indicators is not None:
        return _divide(
            rows, split, lambda frame, share: _take_iterative(frame, indicators.loc[frame.index], share, split.seed)
        )
    strata = _strata(rows[column], split.stratify_bins)
    return _divide(
        rows, split, lambda frame, share: _take_stratified(frame, strata.loc[frame.index], share, split.seed, column)
    )


def _by_groups(rows: Table, split: Split) -> dict[str, Table]:
    column = _column(rows, split.group_by, purpose="group")
    sizes = rows.groupby(column, sort=False).size().sample(frac=1, random_state=split.seed)
    wanted = {name: fraction * len(rows) for name, fraction in split.fractions.items()}
    members: dict[str, list[object]] = {name: [] for name in split.fractions}
    filled = dict.fromkeys(split.fractions, 0)
    for group, size in sizes.items():
        name = max(split.fractions, key=lambda candidate: wanted[candidate] - filled[candidate])
        members[name].append(group)
        filled[name] += int(size)
    parts = {name: rows[rows[column].isin(groups)].reset_index(drop=True) for name, groups in members.items()}
    return _refusing_empty(parts, f"whole groups move together and {column!r} has only {len(sizes)} of them")


def _column(rows: Table, column: str | None, *, purpose: str) -> str:
    if column is None or column not in rows.columns:
        raise KeyError(f"Cannot {purpose} by {column!r}: the table has columns {sorted(map(str, rows.columns))}.")
    return column


def _divide(rows: Table, split: Split, take: Callable[[Table, float], tuple[Table, Table]]) -> dict[str, Table]:
    parts: dict[str, Table] = {}
    names = list(split.fractions)
    remaining, remaining_share = rows, 1.0
    for name in names[:-1]:
        taken, remaining = take(remaining, split.fractions[name] / remaining_share)
        parts[name] = taken.reset_index(drop=True)
        remaining_share -= split.fractions[name]
    parts[names[-1]] = remaining.reset_index(drop=True)
    return _refusing_empty(parts, f"{len(rows)} rows do not stretch across the requested fractions")


def _refusing_empty(parts: dict[str, Table], because: str) -> dict[str, Table]:
    if empty := [name for name, part in parts.items() if part.empty]:
        raise ValueError(f"The split left {', '.join(empty)} without a single row: {because}.")
    return parts


def _label_indicators(column: pd.Series, separator: str) -> Table | None:
    parsed = [labels_in(value, separator) for value in column]
    if all(len(labels) <= 1 for labels in parsed):
        return None
    vocabulary = sorted({label for labels in parsed for label in labels})
    log.info(
        "Stratifying %r by %d labels carried across rows (iterative stratification).", column.name, len(vocabulary)
    )
    return pd.DataFrame(
        {label: [label in labels for labels in parsed] for label in vocabulary}, index=column.index, dtype=int
    )


def _take_iterative(rows: Table, indicators: Table, share: float, seed: int) -> tuple[Table, Table]:
    if share <= 0.0:
        return rows.iloc[:0], rows
    if share >= 1.0:
        return rows, rows.iloc[:0]
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[1.0 - share, share])
        taken, rest = next(stratifier.split(np.zeros((len(rows), 1)), indicators.to_numpy()))
    finally:
        np.random.set_state(state)
    return rows.iloc[taken], rows.iloc[rest]


def _strata(column: pd.Series, bins: int) -> pd.Series:
    distinct = column.nunique(dropna=False)
    if pd.api.types.is_numeric_dtype(column) and distinct > bins:
        quantiles = min(bins, max(2, len(column) // 2))
        binned = pd.qcut(column, q=quantiles, labels=False, duplicates="drop")
        log.info(
            "Stratifying by %d quantile bins of %r (%d distinct values).", int(binned.nunique()), column.name, distinct
        )
        return binned.astype(str)
    log.info("Stratifying by the %d distinct values of %r.", distinct, column.name)
    return column.astype(str)


def _take_stratified(rows: Table, strata: pd.Series, share: float, seed: int, column: str) -> tuple[Table, Table]:
    if share <= 0.0:
        return rows.iloc[:0], rows
    if share >= 1.0:
        return rows, rows.iloc[:0]
    counts = strata.value_counts()
    unsplittable = strata.isin(counts[counts < 2].index)
    kept, splittable = rows[unsplittable], rows[~unsplittable]
    if splittable.empty:
        return kept, splittable
    wanted = max(0, round(share * len(rows)) - len(kept))
    adjusted = min(max(wanted / len(splittable), 1e-9), 1.0 - 1e-9)
    try:
        taken, rest = train_test_split(
            splittable, train_size=adjusted, random_state=seed, stratify=strata.loc[splittable.index]
        )
    except ValueError as error:
        raise ValueError(
            f"Cannot stratify by {column!r}: every split needs one row of each of its {strata.nunique()} values and "
            f"{len(rows)} rows do not stretch that far ({error}). Use a coarser column or drop stratify_by."
        ) from error
    return pd.concat([kept, taken]), rest
