"""What every folder can reach: the marker a folder declares by its name, and one seeded generator.

At the root rather than beside one package, because both halves of the suite need them: an end-to-end
run is marked by where it lives, and a unit test that builds a tensor at import time is as entitled to
a fixed seed as one that builds it inside a function.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

E2E = Path(__file__).parent / "e2e"

SEED = 0
torch.manual_seed(SEED)
"""Seeded here, at import: a conftest runs before the test modules that build tensors at module level.

Nothing in the suite asserts a particular random value, and this is what keeps that true — a run that
fails does so for a reason, not for a draw.
"""


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Every test under ``tests/e2e/`` carries the ``e2e`` marker, by where it lives.

    A marker applied by hand is a marker one file forgets, so the folder is the declaration and this
    is the only place that reads it.
    """
    for item in items:
        if E2E in item.path.parents:
            item.add_marker(pytest.mark.e2e)
