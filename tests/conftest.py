"""Fixtures every folder can reach — the ones a test would otherwise write for itself.

At the root rather than beside one package, because the same dataset serves a build
test and an end-to-end run alike, and a helper only one folder can see is what made eight
copies of it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.datasets import write_dataset

E2E_FOLDER = Path(__file__).parent / "e2e"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Every test under ``tests/e2e/`` carries the ``e2e`` marker, by where it lives.

    A marker applied by hand is a marker one file forgets — measured, one sat on a
    helper rather than its test — so the folder is the declaration and this is the
    only place that reads it.
    """
    for item in items:
        if E2E_FOLDER in item.path.parents:
            item.add_marker(pytest.mark.e2e)


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
