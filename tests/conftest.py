"""Shared pytest fixtures and suite-wide collection hooks.

Fixtures: ``make_image_csv``, a factory for the synthetic image + label CSV that several
test modules built with near-identical private ``csv_path`` fixtures. Each caller passes its own
``count``/``size``/``seed`` (the values its assertions depend on), so the data is byte-identical to
the old inline fixtures — only the duplicated body is gone.

Hooks: every test under ``tests/e2e/`` is auto-marked ``e2e`` (single source of truth — no
per-file ``pytestmark`` boilerplate), so ``-m "not e2e"`` and the ``tests/unit`` path selection
stay equivalent.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

_E2E_DIRECTORY = Path(__file__).parent / "e2e"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every test collected from ``tests/e2e/`` with the ``e2e`` marker."""
    for item in items:
        if _E2E_DIRECTORY in Path(item.fspath).parents:
            item.add_marker(pytest.mark.e2e)


@pytest.fixture
def make_image_csv(tmp_path: Path) -> Callable[..., Path]:
    """Return a factory that writes ``count`` synthetic RGB jpgs and a CSV indexing them.

    Parameters (all keyword, with the historical defaults):
        count (int): Number of images/rows.
        size (int): Square image side in pixels.
        seed (int): NumPy RNG seed — fixes the pixel data (and therefore any stratified split).
        labels (Sequence[str]): Class names cycled by row index (``label`` column).

    Returns:
        Callable[..., Path]: ``make(*, count, size, seed, labels) -> Path`` to the written CSV.
    """

    def _make(
        *,
        count: int = 15,
        size: int = 32,
        seed: int = 0,
        labels: Sequence[str] = ("cat", "dog", "cow"),
    ) -> Path:
        image_dir = tmp_path / "images"
        image_dir.mkdir(exist_ok=True)
        rng = np.random.default_rng(seed)
        rows = []
        for index in range(count):
            array = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
            path = image_dir / f"{index}.jpg"
            cv2.imwrite(str(path), array)
            rows.append({"image_path": str(path), "label": labels[index % len(labels)]})
        csv = tmp_path / "data.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        return csv

    return _make
