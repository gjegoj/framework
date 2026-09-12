"""One table into named splits, by a rule the run declares.

Three rules ship — at random, keeping class shares, keeping whole groups together — and a fourth
arrives by import path with its own options, like a pixel chain or a network. A rule is a callable of
``(rows, fractions, seed)``; everything it needs to know about *how* to divide is its own constructor's
business, which is why no two of them have to be told apart here.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import IterativeStratification

from src.core import validate_name
from src.data.base import Table
from src.data.encoders.label import SEPARATOR, labels_in

log = logging.getLogger(__name__)

type Splitter = Callable[[Table, Mapping[str, float], int], dict[str, Table]]
"""A rule for dividing a table: the rows, the share wanted for each name, and the seed to repeat it by."""


@dataclass(frozen=True, slots=True)
class RandomSplit:
    """Shuffle, then cut: the default, and the only one that needs nothing about the rows."""

    def __call__(self, rows: Table, fractions: Mapping[str, float], seed: int) -> dict[str, Table]:
        shuffled = rows.sample(frac=1, random_state=seed).reset_index(drop=True)
        parts: dict[str, Table] = {}
        names = list(fractions)
        start = 0
        for position, name in enumerate(names):
            end = len(shuffled) if position == len(names) - 1 else start + int(len(shuffled) * fractions[name])
            parts[name] = shuffled.iloc[start:end].reset_index(drop=True)
            start = end
        return _refusing_empty(parts, f"{len(rows)} rows do not stretch across the requested fractions")


@dataclass(frozen=True, slots=True)
class StratifiedSplit:
    """Keep each split's mix of a column the same as the whole table's.

    Parameters:
        by: The column whose shares are kept. Several labels in one cell make it a multilabel
            column, and the division becomes an iterative one over all of them.
        bins: How many quantile bins a numeric column is grouped into before its shares are kept.
        separator: How a cell lists several labels.
    """

    by: str
    bins: int = 10
    separator: str = SEPARATOR

    def __post_init__(self) -> None:
        if self.bins < 2:
            raise ValueError(f"A stratified split needs at least 2 bins, got {self.bins}.")

    def __call__(self, rows: Table, fractions: Mapping[str, float], seed: int) -> dict[str, Table]:
        column = _column(rows, self.by, purpose="stratify")
        indicators = _label_indicators(rows[column], self.separator)
        if indicators is not None:
            return _divide(
                rows, fractions, lambda frame, share: _take_iterative(frame, indicators.loc[frame.index], share, seed)
            )
        strata = _strata(rows[column], self.bins)
        return _divide(
            rows, fractions, lambda frame, share: _take_stratified(frame, strata.loc[frame.index], share, seed, column)
        )


@dataclass(frozen=True, slots=True)
class GroupedSplit:
    """Keep every row of a group on one side, so nothing a group shares leaks between splits.

    A row whose group is missing is refused rather than dropped: ``groupby`` leaves it out of every
    group, and the parts are gathered from the groups, so it would vanish from the run entirely while
    both splits stayed satisfyingly non-empty. Whether unknown patients are one patient or none is a
    question about the data, and it is asked of whoever wrote it.

    Parameters:
        by: The column whose equal values belong together — a patient, a scene, a session.
    """

    by: str

    def __call__(self, rows: Table, fractions: Mapping[str, float], seed: int) -> dict[str, Table]:
        column = _column(rows, self.by, purpose="group")
        blank = rows.index[rows[column].isna()]
        if len(blank):
            raise ValueError(
                f"Cannot group by {column!r}: {len(blank)} rows name no group, the first at {blank[0]}. "
                "Whole groups move together, so a row belonging to none would be dropped from every "
                "split without a word. Give them a group, or divide by a rule that needs none."
            )
        sizes = rows.groupby(column, sort=False).size().sample(frac=1, random_state=seed)
        wanted = {name: fraction * len(rows) for name, fraction in fractions.items()}
        members: dict[str, list[object]] = {name: [] for name in fractions}
        filled = dict.fromkeys(fractions, 0)
        for group, size in sizes.items():
            name = max(fractions, key=lambda candidate: wanted[candidate] - filled[candidate])
            members[name].append(group)
            filled[name] += int(size)
        parts = {name: rows[rows[column].isin(groups)].reset_index(drop=True) for name, groups in members.items()}
        return _refusing_empty(parts, f"whole groups move together and {column!r} has only {len(sizes)} of them")


@dataclass(frozen=True, slots=True)
class Split:
    """How to divide: a fraction per split name, summing to one, and the rule that does the dividing.

    ``seed`` is separate from the experiment's on purpose: runs at different seeds must share
    one test set, or their numbers are not comparable.
    """

    fractions: Mapping[str, float]
    seed: int = 42
    rule: Splitter = field(default_factory=RandomSplit)

    def __post_init__(self) -> None:
        if not self.fractions:
            raise ValueError("A split needs at least one named fraction.")
        for name in self.fractions:
            validate_name(name, label="Split")
        if any(fraction < 0 for fraction in self.fractions.values()):
            raise ValueError("Split fractions are nonnegative.")
        if not math.isclose(sum(self.fractions.values()), 1.0, abs_tol=1e-6):
            # Named, because a run writes the shares flat: a misspelled option reads as one more split.
            raise ValueError(
                f"Split fractions must sum to 1, got {sum(self.fractions.values())} over {', '.join(self.fractions)}."
            )

    @classmethod
    def declared(cls, values: Mapping[str, Any]) -> Self:
        """A split as a run writes it — ``{train: 0.7, val: 0.3, rule: {_target_: …, by: species}}``.

        One flat mapping rather than a nested ``fractions:``, because that is how a division reads: the
        two names below are this class's own options and every other key is a split with its share. The
        cost is that a split cannot be called ``seed`` or ``rule``, and that a misspelled option becomes
        a split — which is why the refusal above names them.
        """
        options = {one.name for one in fields(cls)} - {"fractions"}
        return cls(
            {name: float(value) for name, value in values.items() if name not in options},
            **{name: value for name, value in values.items() if name in options},
        )


def split_table(table: Table, split: Split) -> dict[str, Table]:
    """The rule the split declared, run over the rows: which rule it is was settled when it was built."""
    return split.rule(table.reset_index(drop=True), split.fractions, split.seed)


def _column(rows: Table, column: str | None, *, purpose: str) -> str:
    if column is None or column not in rows.columns:
        raise KeyError(f"Cannot {purpose} by {column!r}: the table has columns {sorted(map(str, rows.columns))}.")
    return column


def _divide(
    rows: Table, fractions: Mapping[str, float], take: Callable[[Table, float], tuple[Table, Table]]
) -> dict[str, Table]:
    parts: dict[str, Table] = {}
    names = list(fractions)
    remaining, remaining_share = rows, 1.0
    for name in names[:-1]:
        taken, remaining = take(remaining, fractions[name] / remaining_share)
        parts[name] = taken.reset_index(drop=True)
        remaining_share -= fractions[name]
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
            f"{len(rows)} rows do not stretch that far ({error}). Use a coarser column, or a rule that needs none."
        ) from error
    return pd.concat([kept, taken]), rest
