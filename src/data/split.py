"""Dataset splitting utilities.

A simple, reproducible random split by ratios. Stratified splitting can be added
later as an alternative strategy without changing callers.
"""

from __future__ import annotations

import pandas as pd

from src.core.enums import Stage

# Deterministic stage order so the remainder always lands on the last present stage.
_STAGE_ORDER = (Stage.TRAIN, Stage.VAL, Stage.TEST)


def split_dataframe(frame: pd.DataFrame, ratios: dict[Stage, float], seed: int) -> dict[Stage, pd.DataFrame]:
    """Shuffle and split ``frame`` into per-stage frames by ``ratios``.

    The last present stage receives the remainder, so every row is assigned and
    rounding never drops samples.

    Parameters:
        frame (pd.DataFrame): Full dataset.
        ratios (dict[Stage, float]): Per-stage ratios (assumed to sum to ~1.0).
        seed (int): Shuffle seed for reproducibility.

    Returns:
        dict[Stage, pd.DataFrame]: Per-stage frames (index reset).
    """
    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    total = len(shuffled)
    present = [stage for stage in _STAGE_ORDER if stage in ratios]

    result: dict[Stage, pd.DataFrame] = {}
    start = 0
    for position, stage in enumerate(present):
        if position == len(present) - 1:
            part = shuffled.iloc[start:]
        else:
            count = int(round(ratios[stage] * total))
            part = shuffled.iloc[start : start + count]
            start += count
        result[stage] = part.reset_index(drop=True)
    return result
