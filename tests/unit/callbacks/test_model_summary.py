"""The model summary's Name column as a tree: the same rows Lightning prints, read as a hierarchy."""

from __future__ import annotations

from typing import Any

import pytest
from lightning.pytorch.callbacks import RichModelSummary

from src.callbacks.model_summary import TreeModelSummary, tree_names
from src.console import HEADER_STYLE


@pytest.mark.parametrize(
    ("paths", "drawn"),
    [
        pytest.param(
            ["learner", "learner.model", "learner.model.backbone", "learner.model.heads"],
            ["learner", "└─ model", "   ├─ backbone", "   └─ heads"],
            id="a branch and its leaves",
        ),
        pytest.param(
            ["model", "model.backbone", "model.backbone.encoder", "model.heads"],
            ["model", "├─ backbone", "│  └─ encoder", "└─ heads"],
            id="depth carries the line down",
        ),
        pytest.param(["model"], ["model"], id="a root stays a name"),
        pytest.param([], [], id="a model summarised as nothing"),
    ],
)
def test_dotted_paths_are_read_as_the_tree_they_describe(paths: list[str], drawn: list[str]) -> None:
    assert tree_names(paths) == drawn


def test_the_rows_keep_the_order_they_were_summarised_in() -> None:
    """Every other column is a parallel list: a row reordered here would take another column's numbers."""
    paths = ["model", "model.heads", "model.backbone"]

    assert [one.rsplit(" ", 1)[-1] for one in tree_names(paths)] == ["model", "heads", "backbone"]


def test_only_the_name_column_and_the_header_are_ours(monkeypatch: pytest.MonkeyPatch) -> None:
    """Columns, totals and rank-zero gating stay Lightning's, so what it adds later arrives for free.

    The header is the exception, and the only one: two more tables are printed under this one, and a
    run reads them as one report. The value is the library's own default today, which is exactly why
    it is said here — otherwise the three agree by coincidence rather than by a line.
    """
    summarised: dict[str, Any] = {}
    monkeypatch.setattr(
        RichModelSummary,
        "summarize",
        staticmethod(lambda *given, **named: summarised.update(given=given, named=named)),
    )

    TreeModelSummary.summarize(
        [("Name", ["model", "model.backbone"]), ("Params", ["10", "10"])], 10, 10, 0.1, {"train": 1}, 0, extra="kept"
    )

    assert summarised["given"][0] == [("Name", ["model", "└─ backbone"]), ("Params", ["10", "10"])]
    assert summarised["given"][1:] == (10, 10, 0.1, {"train": 1}, 0)
    assert summarised["named"] == {"extra": "kept", "header_style": HEADER_STYLE}
