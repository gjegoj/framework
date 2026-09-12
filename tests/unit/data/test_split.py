"""One table becomes named splits; every split keeps at least one row or the run refuses to start."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.split import GroupedSplit, RandomSplit, Split, Splitter, StratifiedSplit, split_table

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
    parts = split_table(table, Split(FRACTIONS, rule=StratifiedSplit(by="species")))

    for part in parts.values():
        assert (part["species"] == "dog").mean() == pytest.approx(0.25, abs=0.05)


def test_stratified_split_bins_a_continuous_column(table: pd.DataFrame) -> None:
    parts = split_table(table, Split(FRACTIONS, rule=StratifiedSplit(by="age", bins=5)))

    assert parts["val"]["age"].mean() == pytest.approx(table["age"].mean(), abs=15)


def test_stratified_split_balances_multilabel_cells_one_label_at_a_time(table: pd.DataFrame) -> None:
    parts = split_table(table, Split(FRACTIONS, rule=StratifiedSplit(by="tags")))

    assert all(part["tags"].str.contains("b").any() for part in parts.values())


def test_group_split_keeps_a_group_in_one_part(table: pd.DataFrame) -> None:
    parts = split_table(table, Split(FRACTIONS, rule=GroupedSplit(by="patient")))

    seen = [set(part["patient"]) for part in parts.values()]
    assert all(a.isdisjoint(b) for i, a in enumerate(seen) for b in seen[i + 1 :])
    assert len(parts["train"]) == pytest.approx(60, abs=10)


RULES = [
    pytest.param(RandomSplit(), id="random"),
    pytest.param(StratifiedSplit(by="species"), id="stratified"),
    pytest.param(GroupedSplit(by="patient"), id="grouped"),
]


@pytest.mark.parametrize("rule", RULES)
def test_every_rule_deals_each_row_exactly_once(rule: Splitter, table: pd.DataFrame) -> None:
    """A division that loses rows is the one failure a split cannot report by being empty.

    Written against the ages, which are unique per row: the parts together have to be the table, and
    no row may sit in two of them.
    """
    parts = split_table(table, Split(FRACTIONS, rule=rule))
    dealt = [age for part in parts.values() for age in part["age"]]

    assert sorted(dealt) == sorted(table["age"]) and len(dealt) == len(set(dealt))


def test_a_row_with_no_group_of_its_own_is_named_rather_than_dropped(table: pd.DataFrame) -> None:
    """Whose patient is unknown is a question for whoever wrote the table: keeping every such row
    together is one experiment and dropping them is another, and pandas answers it silently."""
    holed = table.copy()
    holed.loc[3, "patient"] = None

    with pytest.raises(ValueError, match="patient"):
        split_table(holed, Split(FRACTIONS, rule=GroupedSplit(by="patient")))


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"fractions": {}}, id="no fractions"),
        pytest.param({"fractions": {"train": 0.5, "val": 0.4}}, id="not summing to one"),
        pytest.param({"fractions": {"train": 1.0, "a/b": 0.0}}, id="split name with a separator"),
    ],
)
def test_refuses_a_declaration_it_cannot_serve(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Split(**kwargs)  # type: ignore[arg-type]


def test_a_rule_that_could_never_divide_anything_is_refused_where_it_was_declared() -> None:
    """A rule's own options are refused by the rule, which is the only thing that knows what they mean."""
    with pytest.raises(ValueError, match="bins"):
        StratifiedSplit(by="species", bins=1)


def test_a_missing_column_or_an_empty_part_is_named(table: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="breed"):
        split_table(table, Split(FRACTIONS, rule=StratifiedSplit(by="breed")))
    with pytest.raises(ValueError, match="train"):
        split_table(table.head(1), Split({"train": 0.5, "val": 0.5}))


class TestDeclaration:
    """How a run writes a split: shares by split name, beside the options that shape the division."""

    def test_the_shares_and_the_options_arrive_in_one_flat_mapping(self) -> None:
        rule = StratifiedSplit(by="species")

        declared = Split.declared({"train": 0.7, "val": 0.3, "rule": rule, "seed": 1})

        assert declared == Split({"train": 0.7, "val": 0.3}, seed=1, rule=rule)

    def test_a_share_written_as_a_whole_number_is_still_a_share(self) -> None:
        assert Split.declared({"train": 1}) == Split({"train": 1.0})

    def test_an_option_misspelled_is_named_among_the_splits_it_was_taken_for(self) -> None:
        """The one cost of a flat mapping: a typo becomes a split, so the refusal has to name them."""
        with pytest.raises(ValueError, match="stratify_bin"):
            Split.declared({"train": 0.7, "val": 0.3, "stratify_bin": 5})
