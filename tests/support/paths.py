"""Where the shipped tree is, for the tests that read files rather than import them.

Resolved once, from this file, so a test's depth in the folders says nothing about how far the root is:
moving a test between ``unit/`` and ``e2e/`` must not silently repoint it at a directory that is not there.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
