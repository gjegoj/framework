"""Importing a package is what makes the names its declarations may write resolvable.

Checked in a fresh interpreter on purpose. Inside one test session every module is imported by something,
so a registry that a real run would find empty still looks full here.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
PACKAGES = sorted(path.parent.name for path in (ROOT / "src").glob("*/registry.py") if path.parent.name != "core")
"""Every package that offers names to a declaration; the check grows with the framework."""

SCRIPT = """
import importlib

importlib.import_module("src.{package}")
registry = importlib.import_module("src.{package}.registry")
catalogues = [(name, value) for name, value in vars(registry).items() if name.endswith("_registry")]
print(",".join(sorted(name for name, value in catalogues if not list(value))))
"""


@pytest.mark.parametrize("package", PACKAGES)
def test_importing_a_package_registers_every_name_it_offers(package: str) -> None:
    """A registry left empty here is one whose implementations no `__init__` imports."""
    finished = subprocess.run(
        [sys.executable, "-c", SCRIPT.format(package=package)], capture_output=True, text=True, cwd=ROOT, check=True
    )

    assert finished.stdout.strip() == "", f"{package}: these stayed empty after importing the package"
