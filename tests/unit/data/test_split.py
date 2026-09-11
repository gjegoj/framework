"""One table becomes named splits; every split keeps at least one row or the run refuses to start."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.split import Split, split_table

FRACTIONS = {"train": 0.6, "val": 0.2, "test": 0.2}


@pytest.fixture
def table() -> pd.DataFrame:
    rows = 100
    return pd.DataFrame(
        {
            "species": ["cat" if i % 4 else "dog" for i in range(rows)],  # 75 / 25
            "age": [float(i) for i in range(rows)],
            "tags": ["a,b" if i % 3 == 0 else ("a" if i % 3 == 1 else "b") for i in range(rows)],
            "patient": [i // 5 for i in range(rows)],
        }
    )


def test_random_split_divides_by_fraction_and_is_seeded(table: pd.DataFrame) -> None:
    parts = split_table(table, Split(FRACTIONS, seed=1))
    again = split_table(table, Split(FRACTIONS, seed=1))

    assert {name: len(part) for name, part in parts.items()} == {"train": 60, "val": 20, "test": 20}
    assert sum(len(part) for part in parts.values()) == len(table)
    assert all(parts[name].equals(again[name]) for name in FRACTIONS)
    assert not parts["train"].equals(split_table(table, Split(FRACTIONS, seed=2))["train"])


def test_stratified_split_keeps_class_shares_in_every_part(table: pd.DataFrame) -> None:
    parts = split_table(table, Split(FRACTIONS, stratify_by="species"))

    for part in parts.values():
        assert (part["species"] == "dog").mean() == pytest.approx(0.25, abs=0.05)


def test_stratified_split_bins_a_continuous_column(table: pd.DataFrame) -> None:
    parts = split_table(table, Split(FRACTIONS, stratify_by="age", stratify_bins=5))

    assert parts["val"]["age"].mean() == pytest.approx(table["age"].mean(), abs=15)


def test_stratified_split_balances_multilabel_cells_one_label_at_a_time(table: pd.DataFrame) -> None:
    parts = split_table(table, Split(FRACTIONS, stratify_by="tags"))

    assert all(part["tags"].str.contains("b").any() for part in parts.values())


def test_group_split_keeps_a_group_in_one_part(table: pd.DataFrame) -> None:
    parts = split_table(table, Split(FRACTIONS, group_by="patient"))

    seen = [set(part["patient"]) for part in parts.values()]
    assert all(a.isdisjoint(b) for i, a in enumerate(seen) for b in seen[i + 1 :])
    assert len(parts["train"]) == pytest.approx(60, abs=10)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"fractions": {}}, id="no fractions"),
        pytest.param({"fractions": {"train": 0.5, "val": 0.4}}, id="not summing to one"),
        pytest.param(
            {"fractions": FRACTIONS, "stratify_by": "species", "group_by": "patient"}, id="stratify and group"
        ),
        pytest.param({"fractions": FRACTIONS, "stratify_bins": 1}, id="one bin"),
        pytest.param({"fractions": {"train": 1.0, "a/b": 0.0}}, id="split name with a separator"),
    ],
)
def test_refuses_a_declaration_it_cannot_serve(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Split(**kwargs)  # type: ignore[arg-type]


def test_a_missing_column_or_an_empty_part_is_named(table: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="breed"):
        split_table(table, Split(FRACTIONS, stratify_by="breed"))
    with pytest.raises(ValueError, match="train"):
        split_table(table.head(1), Split({"train": 0.5, "val": 0.5}))
