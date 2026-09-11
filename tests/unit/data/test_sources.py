"""Rows come from files in three formats, one or several per source, and can be capped for a smoke run."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.registry import table_source_registry
from src.data.sources import TableSource, capped, format_of, source_for


@pytest.fixture
def table() -> pd.DataFrame:
    return pd.DataFrame({"image": ["a.png", "b.png", "c.png"], "species": ["cat", "dog", "cat"]})


@pytest.fixture
def files(tmp_path: Path, table: pd.DataFrame) -> dict[str, Path]:
    table.to_csv(tmp_path / "rows.csv", index=False)
    table.to_json(tmp_path / "rows.json", orient="records")
    (tmp_path / "rows.jsonl").write_text("\n".join(json.dumps(row) for row in table.to_dict("records")))
    return {suffix: tmp_path / f"rows.{suffix}" for suffix in ("csv", "json", "jsonl")}


@pytest.mark.parametrize("suffix", ["csv", "json", "jsonl"])
def test_every_format_reads_the_same_rows(files: dict[str, Path], table: pd.DataFrame, suffix: str) -> None:
    read = source_for(files[suffix]).read()

    pd.testing.assert_frame_equal(read, table)


def test_several_files_of_one_format_concatenate_in_order(files: dict[str, Path]) -> None:
    assert len(source_for([files["csv"], files["csv"]]).read()) == 6


def test_the_format_follows_the_suffix_unless_declared(files: dict[str, Path]) -> None:
    assert format_of("x.CSV") == "csv"
    assert isinstance(source_for(files["csv"], format="csv"), TableSource)
    with pytest.raises(LookupError, match="parquet"):
        format_of("x.parquet")


@pytest.mark.parametrize("name", list(table_source_registry))
def test_every_registered_source_is_a_table_source(name: str) -> None:
    assert issubclass(table_source_registry.get(name), TableSource)


class TestCap:
    def test_a_count_or_a_fraction_keeps_a_reproducible_random_subset(self, table: pd.DataFrame) -> None:
        assert len(capped(table, 2)) == 2
        assert len(capped(table, 2 / 3)) == 2
        assert capped(table, 2).equals(capped(table, 2))
        assert capped(table, None) is table

    @pytest.mark.parametrize("cap", [0, -1, 1.5])
    def test_refuses_a_cap_that_selects_nothing_or_more_than_everything(self, table: pd.DataFrame, cap: float) -> None:
        with pytest.raises(ValueError):
            capped(table, cap)
