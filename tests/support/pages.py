"""Trackers a test can look inside: one that shows pages, and one that only holds numbers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lightning.pytorch.loggers import Logger


class NumbersOnly(Logger):
    """What most backends are: a place for scalars, and nothing that renders."""

    @property
    def name(self) -> str:
        return "numbers"

    @property
    def version(self) -> str:
        return "0"

    def log_hyperparams(self, *args: Any, **kwargs: Any) -> None:
        return None

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        return None


class PageRecorder(NumbersOnly):
    """A backend that can show a page, and keeps every one it was given."""

    def __init__(self) -> None:
        super().__init__()
        self.pages: list[tuple[str, str, int]] = []
        self.fails = False
        """Set by a test that needs the far end to be unreachable, which is what a real one may be."""

    @property
    def name(self) -> str:
        return "pages"

    def log_html(self, title: str, html: str, iteration: int) -> None:
        if self.fails:
            raise OSError("the page could not be written")
        self.pages.append((title, html, iteration))
