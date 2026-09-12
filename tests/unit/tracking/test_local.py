"""The backend a run has without a service: what it writes, and where."""

from __future__ import annotations

import json
from pathlib import Path

from src.tracking import KeepsRecord, ShowsPage
from src.tracking.local import PAGES, LocalFiles


def test_it_shows_pages_and_says_so_structurally(tmp_path: Path) -> None:
    """What the samples grid asks of a backend before offering it one; a rename here would be silent."""
    assert isinstance(LocalFiles(save_dir=str(tmp_path)), ShowsPage)


def test_it_keeps_records_and_says_so_structurally(tmp_path: Path) -> None:
    """What a shipped export asks of a backend before offering it one; a rename here would be silent."""
    assert isinstance(LocalFiles(save_dir=str(tmp_path)), KeepsRecord)


def test_the_record_of_what_a_run_produced_lands_beside_the_numbers_as_a_deployment_reads_it(
    tmp_path: Path,
) -> None:
    """A run without a service keeps everything in one directory, and this is the half a deployment reads."""
    tracker = LocalFiles(save_dir=str(tmp_path), version="")

    tracker.log_record("model", {"outputs": [{"name": "species"}]})

    assert json.loads((Path(tracker.log_dir) / "model.json").read_text(encoding="utf-8")) == {
        "outputs": [{"name": "species"}]
    }


def test_a_page_lands_beside_the_numbers_and_needs_nothing_else_to_open(tmp_path: Path) -> None:
    tracker = LocalFiles(save_dir=str(tmp_path), version="")

    tracker.log_html("samples/val", "<html>a page</html>", iteration=3)

    (written,) = (Path(tracker.log_dir) / PAGES).glob("*.html")
    assert written.read_text(encoding="utf-8") == "<html>a page</html>"


def test_the_pages_of_one_run_sort_into_the_order_they_were_drawn(tmp_path: Path) -> None:
    """Ten epochs after two is later, and a name that sorts by string says otherwise."""
    tracker = LocalFiles(save_dir=str(tmp_path), version="")

    for epoch in (2, 10):
        tracker.log_html("samples/val", "<html></html>", iteration=epoch)

    written = sorted(path.name for path in (Path(tracker.log_dir) / PAGES).glob("*.html"))
    assert written == ["samples-val-0002.html", "samples-val-0010.html"]
