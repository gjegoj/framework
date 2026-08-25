"""Table sources: file-backed formats behind the ``TableSource`` port."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import CsvSource, InMemorySource, JsonLinesSource, JsonSource
from src.data.registry import table_source_registry


def write_csv(path: Path, rows: int, sep: str = ",") -> None:
    frame = pd.DataFrame({"path": [f"img_{index}.jpg" for index in range(rows)], "label": ["cat"] * rows})
    frame.to_csv(path, index=False, sep=sep)


def test_csv_source_reads_one_file(tmp_path: Path) -> None:
    write_csv(tmp_path / "annotations.csv", rows=3)

    table = CsvSource(tmp_path / "annotations.csv").read()

    assert len(table) == 3
    assert list(table.columns) == ["path", "label"]


def test_reader_kwargs_forward_to_pandas(tmp_path: Path) -> None:
    write_csv(tmp_path / "semi.csv", rows=2, sep=";")

    table = CsvSource(tmp_path / "semi.csv", sep=";").read()

    assert list(table.columns) == ["path", "label"]


def test_multiple_paths_concatenate_in_order(tmp_path: Path) -> None:
    write_csv(tmp_path / "a.csv", rows=2)
    write_csv(tmp_path / "b.csv", rows=3)

    table = CsvSource([tmp_path / "a.csv", tmp_path / "b.csv"]).read()

    assert len(table) == 5
    assert list(table.index) == [0, 1, 2, 3, 4]


def test_json_source_reads_records(tmp_path: Path) -> None:
    pd.DataFrame({"path": ["a.jpg"], "label": ["cat"]}).to_json(tmp_path / "rows.json", orient="records")

    table = JsonSource(tmp_path / "rows.json").read()

    assert table.iloc[0]["label"] == "cat"


def test_json_lines_reads_one_row_per_line_keeping_nested_objects(tmp_path: Path) -> None:
    """The detection canon's carrier: nested lists arrive as lists, negatives as empty ones."""
    path = tmp_path / "train.jsonl"
    path.write_text(
        '{"image": "a.jpg", "objects": [{"box": [1.0, 2.0, 3.0, 4.0], "class": "dog"}]}\n'
        '{"image": "b.jpg", "objects": []}\n'
    )

    table = JsonLinesSource(path).read()

    assert list(table.columns) == ["image", "objects"]
    assert table["objects"].iloc[0] == [{"box": [1.0, 2.0, 3.0, 4.0], "class": "dog"}]
    assert table["objects"].iloc[1] == []


def test_in_memory_source_returns_the_table_as_is() -> None:
    frame = pd.DataFrame({"x": [1, 2]})

    assert InMemorySource(frame).read() is frame


def test_file_sources_are_registered_for_config() -> None:
    assert set(table_source_registry) == {"csv", "json", "jsonl"}


def test_an_empty_path_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="path"):
        CsvSource([])
