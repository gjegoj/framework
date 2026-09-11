"""Importing a package is what makes the names its declarations may write resolvable.

Checked in a fresh interpreter on purpose. Inside one test session every module is imported by something,
so a registry that a real run would find empty still looks full here.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from tests.support.paths import ROOT

PACKAGES = sorted(path.parent.name for path in (ROOT / "src").glob("*/registry.py") if path.parent.name != "core")
"""Every package that offers names to a declaration; the check grows with the framework."""

SCRIPT = """
import importlib
import pkgutil

package = importlib.import_module("src.{package}")
catalogue = importlib.import_module("src.{package}.registry")
registries = [(name, value) for name, value in vars(catalogue).items() if name.endswith("_registry")]
known = {{name: set(value) for name, value in registries}}
for found in pkgutil.walk_packages(package.__path__, prefix="src.{package}."):
    importlib.import_module(found.name)
empty = [name for name, value in registries if not list(value)]
late = [
    name + ": " + ", ".join(sorted(set(value) - known[name])) for name, value in registries if set(value) - known[name]
]
print("|".join(empty + late))
"""
"""Import the package the way a run does, then import every module in it and see if more names appeared.

The second half is the interesting one: a registry that holds eight of its nine names looks full, and
the ninth fails at build time in whatever run happens to write it.
"""


@pytest.mark.parametrize("package", PACKAGES)
def test_importing_a_package_registers_every_name_it_offers(package: str) -> None:
    """A registry left empty here is one whose implementations no `__init__` imports."""
    finished = subprocess.run(
        [sys.executable, "-c", SCRIPT.format(package=package)], capture_output=True, text=True, cwd=ROOT, check=True
    )

    assert finished.stdout.strip() == "", (
        f"{package}: a registry left empty, or names that only register when a module its `__init__` "
        "does not import runs"
    )
