"""The annotations writer: deterministic rows, one spelling of the row key, honest counts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.converters.annotations import (
    ConversionReport,
    annotation_object,
    annotation_row,
    clipped_box,
    write_annotations,
)


def test_a_file_with_several_clipped_boxes_is_listed_once() -> None:
    """The field says *files*; the per-box count lives in ``clipped``."""
    report = ConversionReport()
    for corners in ((-1.0, 0.0, 5.0, 5.0), (0.0, -2.0, 5.0, 5.0)):
        clipped_box(corners, width=10, height=10, image="a.jpg", report=report)

    assert report.clipped == 2
    assert report.clipped_files == {"a.jpg"}


def test_the_row_key_is_spelled_once_beside_the_object_fields() -> None:
    """The converters write what the readers name; neither spells the key itself."""
    assert annotation_row("x.jpg", []) == {"image": "x.jpg", "objects": []}


def test_a_non_ascii_class_name_survives_the_writer_and_the_reader(tmp_path: Path) -> None:
    """``ensure_ascii=False`` keeps 'собака' readable in review; pinning utf-8 on the
    write is what makes that safe off this machine — ``pd.read_json`` reads utf-8
    unconditionally, so the platform default was the one unpinned link."""
    write_annotations(
        [annotation_row("a.jpg", [annotation_object((1.0, 2.0, 3.0, 4.0), "собака")])], tmp_path / "t.jsonl"
    )

    table = pd.read_json(tmp_path / "t.jsonl", lines=True)

    assert table.loc[0, "objects"][0]["class"] == "собака"
