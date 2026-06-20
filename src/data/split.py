"""Dataset splitting utilities.

Supports random and stratified splits. When ``stratify_column`` is given the
strategy is auto-detected from the column values:

- categorical strings (no commas) — ``sklearn.train_test_split`` with stratify
- numeric values — quantile-binned stratification
- comma-separated strings — ``IterativeStratification`` from scikit-multilearn

Three-way splits (train / val / test) execute two sequential binary splits so
every stage gets a representative label distribution.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import IterativeStratification

from src.core.enums import Stage

_STAGE_ORDER = (Stage.TRAIN, Stage.VAL, Stage.TEST)

_Strategy = Literal["categorical", "numeric", "multilabel"]


def split_dataframe(
    frame: pd.DataFrame,
    ratios: dict[Stage, float],
    seed: int,
    stratify_column: str | None = None,
) -> dict[Stage, pd.DataFrame]:
    """Shuffle and split ``frame`` into per-stage frames by ``ratios``.

    When ``stratify_column`` is given the split is stratified; the strategy is
    auto-detected from the column values. Without it a plain random split is
    performed. The last present stage absorbs the remainder so rounding never
    drops samples.

    Parameters:
        frame (pd.DataFrame): Full dataset.
        ratios (dict[Stage, float]): Per-stage ratios (must sum to ~1.0).
        seed (int): Shuffle seed for reproducibility.
        stratify_column (str | None): Column to stratify by. ``None`` → random.

    Returns:
        dict[Stage, pd.DataFrame]: Per-stage frames with reset indices.
    """
    if stratify_column is not None:
        return _stratified_split(frame, ratios, seed, stratify_column)
    return _random_split(frame, ratios, seed)


# ------------------------------------------------------------------ random


def _random_split(frame: pd.DataFrame, ratios: dict[Stage, float], seed: int) -> dict[Stage, pd.DataFrame]:
    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    total = len(shuffled)
    stages = [stage for stage in _STAGE_ORDER if stage in ratios]

    result: dict[Stage, pd.DataFrame] = {}
    cursor = 0
    for i, stage in enumerate(stages):
        if i == len(stages) - 1:
            part = shuffled.iloc[cursor:]
        else:
            count = int(round(ratios[stage] * total))
            part = shuffled.iloc[cursor : cursor + count]
            cursor += count
        result[stage] = part.reset_index(drop=True)
    return result


# ------------------------------------------------------------------ stratified


def _detect_strategy(series: pd.Series) -> _Strategy:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if series.dropna().astype(str).str.contains(",").any():
        return "multilabel"
    return "categorical"


def _stratified_split(
    frame: pd.DataFrame,
    ratios: dict[Stage, float],
    seed: int,
    stratify_column: str,
) -> dict[Stage, pd.DataFrame]:
    if stratify_column not in frame.columns:
        raise ValueError(f"stratify_column {stratify_column!r} not found in the dataset.")

    strategy = _detect_strategy(frame[stratify_column])
    stages = [stage for stage in _STAGE_ORDER if stage in ratios]

    if len(stages) == 1:
        return {stages[0]: frame.reset_index(drop=True)}

    train_size = ratios[Stage.TRAIN]
    remainder_size = 1.0 - train_size

    train_frame, remainder_frame = _binary_split(frame, train_size, seed, strategy, stratify_column)

    non_train_stages = [stage for stage in stages if stage != Stage.TRAIN]
    if len(non_train_stages) == 1:
        return {
            Stage.TRAIN: train_frame.reset_index(drop=True),
            non_train_stages[0]: remainder_frame.reset_index(drop=True),
        }

    val_size_in_remainder = ratios[Stage.VAL] / remainder_size if remainder_size > 0 else 0.5
    val_frame, test_frame = _binary_split(remainder_frame, val_size_in_remainder, seed, strategy, stratify_column)

    return {
        Stage.TRAIN: train_frame.reset_index(drop=True),
        Stage.VAL: val_frame.reset_index(drop=True),
        Stage.TEST: test_frame.reset_index(drop=True),
    }


def _binary_split(
    frame: pd.DataFrame,
    left_size: float,
    seed: int,
    strategy: _Strategy,
    stratify_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``frame`` into two parts; ``left_size`` is the fraction for the first."""
    if left_size <= 0.0:
        return pd.DataFrame(columns=frame.columns), frame.copy()
    if left_size >= 1.0:
        return frame.copy(), pd.DataFrame(columns=frame.columns)

    if strategy == "multilabel":
        return _multilabel_split(frame, left_size, seed, stratify_column)
    if strategy == "numeric":
        return _numeric_split(frame, left_size, seed, stratify_column)
    return _categorical_split(frame, left_size, seed, stratify_column)


def _categorical_split(
    frame: pd.DataFrame,
    left_size: float,
    seed: int,
    stratify_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left, right = train_test_split(
        frame,
        train_size=left_size,
        random_state=seed,
        stratify=frame[stratify_column],
    )
    return left, right


def _numeric_split(
    frame: pd.DataFrame,
    left_size: float,
    seed: int,
    stratify_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bins = pd.qcut(frame[stratify_column], q=min(5, len(frame) // 2), labels=False, duplicates="drop")
    left, right = train_test_split(frame, train_size=left_size, random_state=seed, stratify=bins)
    return left, right


def _multilabel_split(
    frame: pd.DataFrame,
    left_size: float,
    seed: int,
    stratify_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = _to_multihot(frame[stratify_column])
    # IterativeStratification uses numpy's global random state internally.
    np.random.seed(seed)
    stratifier = IterativeStratification(
        n_splits=2,
        order=2,
        sample_distribution_per_fold=[1.0 - left_size, left_size],
    )
    left_indices, right_indices = next(stratifier.split(np.arange(len(frame)).reshape(-1, 1), labels))
    return frame.iloc[left_indices], frame.iloc[right_indices]


def _to_multihot(series: pd.Series) -> np.ndarray:
    """Convert comma-separated label strings to a ``[N, C]`` multi-hot array."""
    parsed: list[set[str]] = []
    vocab: set[str] = set()
    for value in series:
        labels = {part.strip() for part in str(value).split(",") if part.strip()} if pd.notna(value) else set()
        parsed.append(labels)
        vocab.update(labels)

    ordered = sorted(vocab)
    label_to_column = {label: column for column, label in enumerate(ordered)}
    matrix = np.zeros((len(series), len(ordered)), dtype=int)
    for row, labels in enumerate(parsed):
        for label in labels:
            matrix[row, label_to_column[label]] = 1
    return matrix
