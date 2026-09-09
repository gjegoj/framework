"""The ``e2e`` marker is a fact of the folder, not a decorator anyone remembers to write."""

from __future__ import annotations

import pytest

from tests.conftest import E2E_FOLDER


def test_every_test_in_this_folder_carries_the_e2e_marker(request: pytest.FixtureRequest) -> None:
    """``-m e2e`` selects the whole folder, and ``-m "not e2e"`` leaves none of it behind."""
    collected = [item for item in request.session.items if E2E_FOLDER in item.path.parents]
    unmarked = [item.nodeid for item in collected if item.get_closest_marker("e2e") is None]

    assert collected, "this test collects beside the others"
    assert unmarked == []
