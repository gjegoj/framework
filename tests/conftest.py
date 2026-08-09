"""Fixtures every folder can reach — the ones a test would otherwise write for itself.

At the root rather than beside one package, because the same dataset serves an assembly
test and an end-to-end run alike, and a helper only one folder can see is what made eight
copies of it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.support.datasets import write_dataset

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """A written dataset's root: eight images and the table naming them."""
    write_dataset(tmp_path)
    return tmp_path


@pytest.fixture
def segmentation_root(tmp_path: Path) -> Path:
    """The same, plus one index-map mask per row under a ``mask`` column."""
    write_dataset(tmp_path, masks=True)
    return tmp_path
