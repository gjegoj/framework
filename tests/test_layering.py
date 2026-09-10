"""The dependency rule as a test: arrows point down only, and every third-party stack has one home.

A rule stated in prose holds until the first convenient import. This reads every import
under ``src/`` (nested and lazy imports included) and names the file that broke a rule.
Every rule below is a decision; a new arrow is recorded here, never discovered later.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import pytest

SRC = Path(__file__).parents[1] / "src"

QUARANTINE: dict[str, tuple[str, ...]] = {
    "lightning": ("training/", "callbacks/", "tracking/", "build.py", "experiment.py"),
    "lightning_utilities": (),
    "pydantic": ("config/",),
    "hydra": ("cli.py", "config/instantiate.py"),
    "omegaconf": ("cli.py",),
    "albumentations": ("transforms/",),
    "albucore": ("transforms/",),
    "torchvision": ("transforms/",),
    "timm": ("models/backbones/",),
    "transformers": ("models/backbones/",),
    "ultralytics": ("models/backbones/",),
    "peft": ("models/",),
    "segmentation_models_pytorch": ("models/backbones/", "losses/segmentation.py"),
    "torchmetrics": ("metrics/",),
    "clearml": ("tracking/",),
    "plotly": ("tracking/",),
    "cv2": ("data/",),
    "pandas": ("data/",),
    "sklearn": ("data/",),
    "skmultilearn": ("data/",),
    "onnx": ("export/",),
    "onnxruntime": ("export/",),
    "onnxscript": ("export/",),
    "onnxsim": ("export/",),
    "tensorrt": ("export/",),
    "rich": ("console.py", "progress.py", "cli.py", "callbacks/", "export/verification.py"),
}
"""Library → the paths under ``src/`` allowed to import it; an empty tuple bans it outright."""

CORE_MAY_IMPORT = ("torch", "src.core")
"""Besides the standard library: ``core/`` is the vocabulary every package speaks, so it knows no package."""

CAPABILITY_EDGES: dict[str, frozenset[str]] = {
    "callbacks": frozenset(
        {"console", "data", "tracking", "models", "tasks", "training", "transforms", "visualization"}
    ),
    "data": frozenset({"progress", "transforms"}),
    "export": frozenset({"console", "models", "tasks"}),
    "inference": frozenset({"data", "models", "tasks"}),
    "metrics": frozenset({"tracking"}),
    "tasks": frozenset(),
    "training": frozenset({"data", "tracking", "metrics", "tasks", "models", "losses"}),
    "transforms": frozenset({"tasks"}),
    "tracking": frozenset(),
    "losses": frozenset(),
    "models": frozenset(),
    "visualization": frozenset(),
    "integrations": frozenset({"visualization"}),
    "config": frozenset(),
}
"""Which capability may import which, besides ``core`` (and ``config`` from a build module)."""

CONFIG_READERS = ("build.py", "cli.py", "experiment.py")
"""Only the composition root and a package's own ``build.py`` read declarations."""

TRAINING_MODULE_MAY_IMPORT = ("src.core", "src.training", "src.metrics.base", "src.tracking.report")
"""``training/module.py`` asks metric sets and trackers what their contracts promise, never how they are built."""


class Import(NamedTuple):
    file: str
    module: str

    @property
    def library(self) -> str:
        return self.module.split(".")[0]

    @property
    def package(self) -> str:
        """The capability a file belongs to; a top-level module is its own package."""
        return self.file.split("/")[0].removesuffix(".py")

    @property
    def target(self) -> str | None:
        """The capability a ``src.*`` import reaches, or None for anything else."""
        parts = self.module.split(".")
        return parts[1] if parts[0] == "src" and len(parts) > 1 else None


def imports_of(path: Path) -> Iterable[str]:
    """Every module a file imports, nested imports included: a lazy import is still a dependency."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


@pytest.fixture(scope="module")
def imports() -> list[Import]:
    return [
        Import(path.relative_to(SRC).as_posix(), module)
        for path in sorted(SRC.rglob("*.py"))
        for module in imports_of(path)
    ]


@pytest.fixture(scope="module")
def files() -> list[str]:
    return [path.relative_to(SRC).as_posix() for path in sorted(SRC.rglob("*.py"))]


MINIMUM_IMPORTS = 0
"""Raised to 100 once the composition root is wired; until then the tree is legitimately small."""


def test_the_tree_is_read(files: list[str], imports: list[Import]) -> None:
    """A glob that matched nothing would make every rule below vacuous."""
    assert files and len(imports) >= MINIMUM_IMPORTS


@pytest.mark.parametrize(("library", "homes"), QUARANTINE.items(), ids=QUARANTINE)
def test_a_quarantined_library_is_imported_only_from_its_home(
    imports: list[Import], library: str, homes: tuple[str, ...]
) -> None:
    strays = sorted(
        f"{one.file} imports {one.module}"
        for one in imports
        if one.library == library and not one.file.startswith(homes)
    )

    assert strays == []


def test_core_imports_torch_the_standard_library_and_itself_only(imports: list[Import]) -> None:
    reaching_out = sorted(
        f"{one.file} imports {one.module}"
        for one in imports
        if one.file.startswith("core/")
        and one.library not in sys.stdlib_module_names
        and not one.module.startswith(CORE_MAY_IMPORT)
    )

    assert reaching_out == []


DEBT = {
    "tasks": "phase 5: Task stops owning its loss, LossInput moves to core",
}
"""Edges the v2 skeleton still crosses; a strict xfail turns green the day the debt is paid."""


@pytest.mark.parametrize(
    ("package", "allowed"),
    [
        pytest.param(
            package,
            allowed,
            id=package,
            marks=pytest.mark.xfail(strict=True, reason=DEBT[package]) if package in DEBT else (),
        )
        for package, allowed in CAPABILITY_EDGES.items()
    ],
)
def test_a_capability_consumes_others_only_along_its_declared_edges(
    imports: list[Import], package: str, allowed: frozenset[str]
) -> None:
    undeclared = sorted(
        f"{one.file} imports {one.module}"
        for one in imports
        if one.package == package and one.target is not None and one.target not in {package, "core", "config"} | allowed
    )

    assert undeclared == []


def test_only_build_modules_read_config(imports: list[Import]) -> None:
    """A capability receives ready objects; the declaration is read once, at the root or in its own build."""
    readers = sorted(
        one.file
        for one in imports
        if one.target == "config" and not one.file.startswith("config/") and Path(one.file).name not in CONFIG_READERS
    )

    assert readers == []


def test_the_training_module_reads_capabilities_through_their_contracts_only(imports: list[Import]) -> None:
    reaching_in = sorted(
        one.module
        for one in imports
        if one.file == "training/module.py"
        and one.target is not None
        and not one.module.startswith(TRAINING_MODULE_MAY_IMPORT)
    )

    assert reaching_in == []


def test_visualization_is_a_library_of_its_own(imports: list[Import]) -> None:
    """Display types and renderers know nothing of the framework; ``integrations`` converts results into them."""
    reaching_out = sorted(
        f"{one.file} imports {one.module}"
        for one in imports
        if one.file.startswith("visualization/") and one.target not in {None, "visualization"}
    )

    assert reaching_out == []
