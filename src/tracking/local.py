"""The backend a run has without a service: numbers in a file, and the pages that go beside them."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from src.tracking.registry import tracker_registry

PAGES = "pages"
"""The folder a run's pages land in, under wherever the numbers went."""


@tracker_registry.register("csv")
class LocalFiles(CSVLogger):
    """Lightning's CSV logger, and the one thing a run without a service otherwise cannot keep.

    A page carries everything it needs, so a file on disk shows the same thing a hosted panel would —
    which is the difference between a sample grid being a feature of this framework and a feature of
    whoever is paying for a tracker. Numbers stay exactly as the library writes them.
    """

    @rank_zero_only
    def log_html(self, title: str, html: str, iteration: int) -> None:
        """The ``ShowsPage`` port: one file per page, named so a run's pages sort into reading order.

        The title's own separator becomes a dash rather than a folder: a page is named after what it
        shows (``samples/val``), and letting that dig a directory per stage would scatter four files
        over three folders for no one's benefit.
        """
        directory = Path(self.log_dir) / PAGES
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{title.replace('/', '-')}-{iteration:04d}.html").write_text(html, encoding="utf-8")

    @rank_zero_only
    def log_record(self, name: str, record: Mapping[str, object]) -> None:
        """The ``KeepsRecord`` port: the record beside the numbers, in the form its readers take.

        A run without a service keeps everything in one directory, and this is the half of it a
        deployment reads — the same one the export writes next to the artifacts themselves, here
        because a run's own directory is where somebody looks for what the run produced.
        """
        directory = Path(self.log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{name}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
