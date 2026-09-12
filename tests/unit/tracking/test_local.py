"""The backend a run has without a service: what it writes, and where."""

from __future__ import annotations

from pathlib import Path

from src.tracking import ShowsPage
from src.tracking.local import PAGES, LocalFiles


def test_it_shows_pages_and_says_so_structurally(tmp_path: Path) -> None:
    """What the samples grid asks of a backend before offering it one; a rename here would be silent."""
    assert isinstance(LocalFiles(save_dir=str(tmp_path)), ShowsPage)


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
