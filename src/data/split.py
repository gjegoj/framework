"""Stage splitting: one annotation table in, one table per stage out."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import IterativeStratification

from src.core.taxonomy import Stage
from src.data.sources import Table

log = logging.getLogger(__name__)

type Splitter = Callable[[Table], dict[Stage, Table]]
"""Splits the full annotation table into per-stage tables."""


def random_split(fractions: Mapping[Stage, float], seed: int) -> Splitter:
    """Build a splitter that shuffles rows and cuts them by ``fractions``.

    Fractions must sum to 1. Cut sizes are floored, and the last stage absorbs
    the rounding remainder, so every row lands in exactly one stage.

    Parameters:
        fractions (Mapping[Stage, float]): Per-stage share of rows, summing to 1.
        seed (int): Shuffle seed; the same seed always yields the same split.
    """
    _validate_fractions(fractions, caller="random_split")

    def split(table: Table) -> dict[Stage, Table]:
        shuffled = table.sample(frac=1, random_state=seed).reset_index(drop=True)
        parts: dict[Stage, Table] = {}
        stages = list(fractions)
        start = 0
        for position, stage in enumerate(stages):
            is_last = position == len(stages) - 1
            end = len(shuffled) if is_last else start + int(len(shuffled) * fractions[stage])
            parts[stage] = shuffled.iloc[start:end].reset_index(drop=True)
            start = end
        return parts

    return split


def stratified_split(
    fractions: Mapping[Stage, float],
    by: str,
    seed: int,
    bins: int = 10,
    separator: str = ",",
) -> Splitter:
    """Build a splitter that gives every stage the same distribution of ``by``.

    A plain random split leaves stage composition to chance: with an imbalanced
    target, a small validation set can end up with too few — or zero — rows of the
    rare class, and its metrics then say more about the draw than about the model.

    How a row is grouped follows from the column's *content*, not its dtype:
    values that repeat (class labels, whether stored as text or as integers) group
    by the value itself, while a numeric column with more distinct values than
    ``bins`` groups by quantile. Deciding on dtype instead would quantile-bin
    integer class labels and collapse an imbalanced 0/1 target into one bin,
    degrading the split to a random one without any error.

    Cells carrying several labels at once ("cat,dog", or a list) are balanced one
    label at a time by iterative stratification: past a handful of labels their
    combinations are nearly unique, and holding combinations proportional would
    leave almost every row unsplittable.

    Values too rare to spread across stages join the earliest stage that claims
    them, so long-tail data stays usable instead of failing the run.

    Parameters:
        fractions (Mapping[Stage, float]): Per-stage share of rows, summing to 1.
        by (str): Column whose distribution is held equal across stages.
        seed (int): Split seed; the same seed always yields the same split.
        bins (int): Quantile count used for continuous columns.
        separator (str): Separator splitting a multi-label cell into labels.
    """
    _validate_fractions(fractions, caller="stratified_split")
    if bins < 2:
        raise ValueError(f"stratified_split needs at least 2 bins, got {bins}.")

    def split(table: Table) -> dict[Stage, Table]:
        rows = _rows_with_column(table, by, purpose="stratify")
        indicators = _label_indicators(rows[by], separator)
        if indicators is not None:

            def take_iterative(frame: Table, share: float) -> tuple[Table, Table]:
                return _take_iterative(frame, indicators.loc[frame.index], share, seed)

            return _divide(rows, fractions, take_iterative)

        strata = _strata(rows[by], bins)

        def take(frame: Table, share: float) -> tuple[Table, Table]:
            return _take_stratified(frame, strata.loc[frame.index], share, seed, by)

        return _divide(rows, fractions, take)

    return split


def group_split(fractions: Mapping[Stage, float], by: str, seed: int) -> Splitter:
    """Build a splitter that keeps rows sharing a value of ``by`` in one stage.

    Rows are often not independent: several scans of one patient, frames of one
    video, crops of one image. A row-wise split scatters such a family across
    stages, and the model then meets in test what it already memorised in train —
    the metric measures recall of a specific patient, not generalisation.

    Whole groups move together, so stage sizes approximate the fractions instead
    of matching them exactly; that slack is inherent to keeping groups intact.
    The fractions still count *rows*, as they do everywhere else here — which is
    why sklearn's ``GroupShuffleSplit`` is not used: its ``train_size`` is a share
    of groups, so with groups of unequal size (the ordinary case, patients
    contributing different numbers of scans) asking for 0.6 can hand over a fifth
    of the data.

    Parameters:
        fractions (Mapping[Stage, float]): Per-stage share of rows, summing to 1.
        by (str): Column identifying the group a row belongs to.
        seed (int): Split seed; the same seed always yields the same split.
    """
    _validate_fractions(fractions, caller="group_split")

    def split(table: Table) -> dict[Stage, Table]:
        rows = _rows_with_column(table, by, purpose="group")
        sizes = rows.groupby(by, sort=False).size().sample(frac=1, random_state=seed)
        log.info("Splitting by whole groups: %d distinct values of '%s'.", len(sizes), by)

        targets = {stage: fraction * len(rows) for stage, fraction in fractions.items()}
        members: dict[Stage, list[Any]] = {stage: [] for stage in fractions}
        filled = dict.fromkeys(fractions, 0)
        for group, size in sizes.items():
            # Every stage is served at once rather than peeled off one by one: filling train
            # first lets it take the groups a later stage needed and leaves that stage empty.
            stage = max(fractions, key=lambda candidate: targets[candidate] - filled[candidate])
            members[stage].append(group)
            filled[stage] += int(size)

        parts = {stage: rows[rows[by].isin(groups)].reset_index(drop=True) for stage, groups in members.items()}
        if empty := [str(stage) for stage, part in parts.items() if part.empty]:
            raise ValueError(
                f"The split left {', '.join(empty)} without a single row: whole groups move "
                f"together, and '{by}' has only {len(sizes)} of them. Use a finer grouping column, "
                f"or drop 'group_by' if the rows are independent."
            )
        return parts

    return split


def _rows_with_column(table: Table, column: str, purpose: str) -> Table:
    """The table re-indexed for positional work, once the column it is split by exists."""
    if column not in table.columns:
        raise KeyError(
            f"Cannot {purpose} by '{column}': the annotation table has no such column. "
            f"Available columns: {sorted(map(str, table.columns))}."
        )
    return table.reset_index(drop=True)


def _divide(
    rows: Table,
    fractions: Mapping[Stage, float],
    take: Callable[[Table, float], tuple[Table, Table]],
) -> dict[Stage, Table]:
    """Peel one stage at a time off the rows that are left, ``take`` deciding which ones."""
    parts: dict[Stage, Table] = {}
    stages = list(fractions)
    remaining, remaining_share = rows, 1.0
    for stage in stages[:-1]:
        taken, remaining = take(remaining, fractions[stage] / remaining_share)
        parts[stage] = taken.reset_index(drop=True)
        remaining_share -= fractions[stage]
    parts[stages[-1]] = remaining.reset_index(drop=True)

    if empty := [str(stage) for stage, part in parts.items() if part.empty]:
        raise ValueError(
            f"The split left {', '.join(empty)} without a single row, so those stages would report "
            f"nothing. {len(rows)} rows do not stretch across the requested fractions."
        )
    return parts


def _validate_fractions(fractions: Mapping[Stage, float], caller: str) -> None:
    if not fractions:
        raise ValueError(f"{caller} needs a non-empty fractions mapping.")
    total = sum(fractions.values())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"Fractions must sum to 1, got {total}.")


def _label_indicators(column: pd.Series, separator: str) -> Table | None:
    """The ``[rows, labels]`` indicator frame when cells carry several labels, else ``None``.

    ``None`` means one label per row at most, where balancing the values
    themselves is exact and no approximation is called for.
    """
    parsed = [_labels_in(value, separator) for value in column]
    if all(len(labels) <= 1 for labels in parsed):
        return None
    vocabulary = sorted({label for labels in parsed for label in labels})
    log.info(
        "Stratifying '%s' by %d labels carried across rows (iterative stratification).",
        column.name,
        len(vocabulary),
    )
    return pd.DataFrame(
        {label: [label in labels for labels in parsed] for label in vocabulary},
        index=column.index,
        dtype=int,
    )


def _labels_in(value: Any, separator: str) -> set[str]:
    """The labels one cell carries, in either of the two forms a table stores them."""
    if isinstance(value, list | tuple | set):
        return {str(item).strip() for item in value if str(item).strip()}
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    return {part.strip() for part in str(value).split(separator) if part.strip()}


def _take_iterative(rows: Table, indicators: Table, share: float, seed: int) -> tuple[Table, Table]:
    """Take ``share`` of ``rows`` while holding every label's rate steady, not every combination's.

    With more than a handful of labels their combinations are nearly unique, so
    treating a combination as a class leaves almost every row unsplittable.
    Iterative stratification balances one label at a time instead, rarest first.
    """
    if share <= 0.0:
        return rows.iloc[:0], rows
    if share >= 1.0:
        return rows, rows.iloc[:0]

    # IterativeStratification takes no random_state and draws from numpy's global RNG.
    # Seed it for a reproducible split, then hand the caller's state back: borrowing
    # global randomness must leave no trace on the rest of the run.
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        stratifier = IterativeStratification(
            n_splits=2,
            order=2,
            sample_distribution_per_fold=[1.0 - share, share],
        )
        taken, rest = next(stratifier.split(np.zeros((len(rows), 1)), indicators.to_numpy()))
    finally:
        np.random.set_state(state)
    return rows.iloc[taken], rows.iloc[rest]


def _strata(column: pd.Series, bins: int) -> pd.Series:
    """The group each row is balanced within: its own value, or its quantile bin."""
    distinct = column.nunique(dropna=False)
    if pd.api.types.is_numeric_dtype(column) and distinct > bins:
        quantiles = min(bins, max(2, len(column) // 2))
        binned = pd.qcut(column, q=quantiles, labels=False, duplicates="drop")
        log.info(
            "Stratifying by %d quantile bins of '%s' (%d distinct values, treated as continuous).",
            int(binned.nunique()),
            column.name,
            distinct,
        )
        return binned.astype(str)
    log.info("Stratifying by the %d distinct values of '%s' (treated as classes).", distinct, column.name)
    return column.astype(str)


def _take_stratified(rows: Table, strata: pd.Series, share: float, seed: int, column: str) -> tuple[Table, Table]:
    """Take ``share`` of ``rows`` while keeping every stratum's proportion intact."""
    if share <= 0.0:
        return rows.iloc[:0], rows
    if share >= 1.0:
        return rows, rows.iloc[:0]

    counts = strata.value_counts()
    unsplittable = strata.isin(counts[counts < 2].index)
    if unsplittable.any():
        log.info(
            "%d row(s) hold a value of '%s' that occurs once in the rows left to split; "
            "a single row cannot be spread across stages, so they join this stage.",
            int(unsplittable.sum()),
            column,
        )
    kept, splittable = rows[unsplittable], rows[~unsplittable]
    if splittable.empty:
        return kept, splittable

    wanted = max(0, round(share * len(rows)) - len(kept))
    adjusted = min(max(wanted / len(splittable), 1e-9), 1.0 - 1e-9)
    try:
        taken, rest = train_test_split(
            splittable,
            train_size=adjusted,
            random_state=seed,
            stratify=strata.loc[splittable.index],
        )
    except ValueError as error:
        raise ValueError(
            f"Cannot stratify by '{column}': every stage needs at least one row of each of its "
            f"{strata.nunique()} distinct values, and {len(rows)} rows do not stretch that far "
            f"(pandas: {error}). Use a coarser column, merge rare values, or drop 'stratify_by' "
            f"to fall back on random_split."
        ) from error
    return pd.concat([kept, taken]), rest
