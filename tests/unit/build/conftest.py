"""The smallest complete declaration: what a shipped config says, spelled out rather than composed."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from tests.support.declarations import SIZE, smallest_run
from tests.support.table import write_table

__all__ = ["SIZE"]


@pytest.fixture(scope="module")
def table(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_table(tmp_path_factory.mktemp("rows"))


@pytest.fixture
def declaration(table: Path, tmp_path: Path) -> Mapping[str, Any]:
    """One classification task over eight images, trained for a single step on the processor."""
    return smallest_run(table, tmp_path / "run")
