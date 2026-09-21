"""The tools beside the framework, invoked the way a reader invokes them.

As a subprocess and as a module, because that is the command the documentation gives and the only one
that works; why it is the only one is written where the decision lives, in ``scripts/cut_head_tail.py``.
"""

from __future__ import annotations

import subprocess
import sys

from tests.support.paths import ROOT


def cut_head_tail(*arguments: str) -> subprocess.CompletedProcess[str]:
    """``scripts/cut_head_tail.py``, in an environment that grants it nothing it would not have."""
    return subprocess.run(
        [sys.executable, "-m", "scripts.cut_head_tail", *arguments],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
