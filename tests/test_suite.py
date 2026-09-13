"""The rule about the suite itself: what a gate does not collect, it does not check.

Beside ``test_layering`` and ``test_registries`` rather than under a package, because this is about the
whole tree and about the run that reads it, not about anything in ``src``.
"""

from __future__ import annotations

from fnmatch import fnmatch

import pytest

from tests.support.paths import ROOT

TESTS = ROOT / "tests"


def test_no_folder_of_tests_is_one_pytest_walks_past(pytestconfig: pytest.Config) -> None:
    """A folder whose name matches ``norecursedirs`` is skipped in silence — no error, no skip, no line.

    This would be untrue if a folder of tests were invisible to every Makefile target while the report
    said the suite was green. Measured before this existed: pytest's own default list carries the
    pattern ``build``, so ``tests/unit/build`` — the composition root's own 43 tests — was collected by
    ``make test``, ``make test-unit`` and ``make test-gate`` alike, which is to say by none of them.
    """
    patterns = list(pytestconfig.getini("norecursedirs"))
    walked_past = sorted(
        str(folder.relative_to(ROOT))
        for folder in TESTS.rglob("*")
        if folder.is_dir() and any(folder.rglob("test_*.py")) and any(fnmatch(folder.name, one) for one in patterns)
    )

    assert not walked_past, (
        f"These hold tests and are named so that pytest never enters them: {', '.join(walked_past)}. "
        f"The patterns it walks past are {', '.join(patterns)} — rename the folder, or take the pattern "
        "out of `norecursedirs` in pyproject.toml."
    )
