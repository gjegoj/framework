"""Group splitting: rows that belong together are never separated by the split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core import Stage
from src.data import group_split

FRACTIONS = {Stage.TRAIN: 0.6, Stage.VAL: 0.2, Stage.TEST: 0.2}


def scans(patients: int = 50, per_patient: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient": np.repeat([f"p{index}" for index in range(patients)], per_patient),
            "image": [f"{index}.png" for index in range(patients * per_patient)],
        }
    )


def test_no_group_is_split_between_stages() -> None:
    """The point of the whole feature: one patient's scans cannot sit in train and in test."""
    parts = group_split(FRACTIONS, by="patient", seed=42)(scans())

    seen: dict[str, Stage] = {}
    for stage, part in parts.items():
        for patient in part["patient"]:
            assert seen.setdefault(patient, stage) == stage, f"'{patient}' appears in {seen[patient]} and {stage}"


def test_every_row_lands_in_exactly_one_stage() -> None:
    parts = group_split(FRACTIONS, by="patient", seed=42)(scans())

    assigned = sorted(image for part in parts.values() for image in part["image"])
    assert assigned == sorted(scans()["image"])


def test_stage_sizes_stay_close_to_the_requested_fractions() -> None:
    """Whole groups move together, so sizes approximate the fractions rather than hit them."""
    parts = group_split(FRACTIONS, by="patient", seed=42)(scans())

    assert abs(len(parts[Stage.TRAIN]) / 200 - 0.6) < 0.1


def test_fractions_count_rows_not_groups() -> None:
    """Groups differ in size, and '0.6' has to keep meaning 60% of the data.

    sklearn's ``GroupShuffleSplit`` reads its ``train_size`` as a share of groups,
    so on this table asking for 0.6 hands over 60% of the groups — a fifth of the rows.
    """
    table = pd.DataFrame(
        {
            "group": ["big"] * 40 + ["s1"] * 20 + ["s2"] * 20 + ["s3"] * 10 + ["s4"] * 10,
            "row": range(100),
        }
    )

    parts = group_split(FRACTIONS, by="group", seed=42)(table)

    assert len(parts[Stage.TRAIN]) >= 50
    assert sum(len(part) for part in parts.values()) == 100


def test_the_same_seed_gives_the_same_split() -> None:
    split = group_split(FRACTIONS, by="patient", seed=42)

    assert split(scans())[Stage.TEST]["image"].tolist() == split(scans())[Stage.TEST]["image"].tolist()


def test_an_unknown_column_names_the_available_ones() -> None:
    with pytest.raises(KeyError, match="hospital") as caught:
        group_split(FRACTIONS, by="hospital", seed=42)(scans())

    assert "patient" in str(caught.value)  # the available columns, so the typo is findable


def test_groups_too_coarse_for_the_fractions_are_reported() -> None:
    """Three patients cannot fill three stages; failing loudly beats an empty test set."""
    table = pd.DataFrame({"patient": np.repeat(["a", "b", "c"], 10), "image": range(30)})

    with pytest.raises(ValueError, match="group"):
        group_split(FRACTIONS, by="patient", seed=42)(table)
