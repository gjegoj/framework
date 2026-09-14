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
    "lightning": ("training/", "callbacks/", "tracking/", "build.py", "experiment.py", "cli.py"),
    "lightning_utilities": (),
    "pydantic": ("config/",),
    "hydra": ("config/instantiate.py", "cli.py"),
    "omegaconf": ("cli.py",),
    "albumentations": ("transforms/albumentations.py", "transforms/augmentations/"),
    "peft": ("models/adapters.py",),
    "timm": ("models/backbones/",),
    "segmentation_models_pytorch": ("models/backbones/", "losses/segmentation.py"),
    "torchmetrics": ("metrics/",),
    "clearml": ("tracking/",),
    "ncnn": ("export/backends/ncnn.py",),
    "pnnx": ("export/backends/ncnn.py",),
    "onnx": ("export/backends/onnx.py",),
    "onnxruntime": ("export/backends/onnx.py",),
    "onnxsim": ("export/backends/onnx.py",),
    "tensorrt": ("export/backends/tensorrt.py",),
    "cv2": ("data/",),
    "pandas": ("data/",),
    "sklearn": ("data/",),
    "skmultilearn": ("data/",),
    "rich": ("console.py", "callbacks/", "cli.py"),
}
"""Library → the paths under ``src/`` allowed to import it; an empty tuple bans it outright.

Every home names a path that exists, and every permission names a library the tree actually imports, so
a row written ahead of its code cannot sit here looking like a rule while holding nothing: the row
arrives with what it permits. A ban is exempt from the second half — its whole point is to hold for
something absent.
"""

CORE_MAY_IMPORT = ("torch", "src.core")
"""Besides the standard library: ``core/`` is the vocabulary every package speaks, so it knows no package."""

CAPABILITY_EDGES: dict[str, frozenset[str]] = {
    "build": frozenset(
        {
            "callbacks",
            "data",
            "experiment",
            "export",
            "losses",
            "metrics",
            "models",
            "tasks",
            "tracking",
            "training",
            "transforms",
        }
    ),
    "cli": frozenset({"build", "console", "experiment", "export"}),
    # `models`, because folding a delta back in is a step of the run, between what it kept and what it ships.
    "experiment": frozenset({"export", "models", "tracking", "training"}),
    # `models` and `tasks` for what a deployable graph is made of: one network, and what its outputs mean.
    "export": frozenset({"models", "tasks"}),
    # `losses`, because annealing moves a number of an objective and has to know what one is.
    # `console`, because a callback that prints a table prints through the one terminal everything shares.
    "callbacks": frozenset(
        {"console", "integrations", "losses", "tracking", "training", "transforms", "visualization"}
    ),
    "data": frozenset({"console", "transforms"}),
    "metrics": frozenset(),
    "tasks": frozenset(),
    "training": frozenset({"data", "tracking", "metrics", "tasks", "models", "losses"}),
    "transforms": frozenset({"tasks"}),
    "tracking": frozenset(),
    "losses": frozenset(),
    "models": frozenset(),
    "config": frozenset(),
    "console": frozenset(),
    "integrations": frozenset({"tasks", "visualization"}),
    "visualization": frozenset(),
}
"""Which capability may import which, besides ``core`` (and ``config`` from a build module)."""

PACKAGES = sorted(path.name for path in SRC.iterdir() if (path / "__init__.py").exists() and path.name != "core")
"""Every package under ``src/``; a new one is unconstrained until it appears in the table above."""

CONFIG_READERS = ("build.py", "cli.py", "experiment.py")
"""Only the composition root and a package's own ``build.py`` read declarations."""

VISUALIZATION = "visualization/"
VISUALIZATION_MAY_IMPORT = ("numpy",)
"""What the one package meant to leave this tree may reach for, besides the standard library.

An empty edge set would still let it import ``core``, and a package carrying this framework's entities
is not a library anyone else can use. The rule here is the stronger one it is written to: plain values
over numpy, and nothing of ours at all.
"""

TRAINING_MODULE = "training/module.py"
TRAINING_MODULE_FACADES = ("src.core", "src.metrics", "src.tracking")
"""The loop asks metrics and tracking for their contracts, through the one door each package publishes.

Facades only, and matched exactly: what a metric collection *is* and where a value goes are contracts,
while which backend draws it and how one is built are not this module's business. Read as prefixes
these three would have admitted `src.metrics.classification` and `src.tracking.clearml` — the two
imports the rule exists to keep out — so the promise was the docstring's alone. Its own package is
exempt: reaching into a neighbour is what this forbids, and `src.training.base` is not one."""


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
    """Every module a file imports — nested, lazy and relative alike; each of them is a dependency.

    A relative import is resolved to the name it actually reaches, because that is the only form the
    rules below read. Skipping them made every rule hold for one spelling of an import and not the
    other, and rewriting a package's imports as relative is the first step of lifting it out — which
    is the operation ``test_visualization_is_a_library_of_its_own`` exists to protect.
    """
    here = ("src", *path.relative_to(SRC).parts[:-1])
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and (reached := _reaches(here, node)):
            yield reached


def _reaches(here: tuple[str, ...], node: ast.ImportFrom) -> str:
    """What a ``from`` import names, written absolute or relative to the package it sits in."""
    if not node.level:
        return node.module or ""
    root = here[: len(here) - node.level + 1]
    return ".".join((*root, *((node.module,) if node.module else ())))


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


MINIMUM_IMPORTS = 950
"""What the tree imports today, rounded down.

The rules below all read the same list, so a glob that quietly stopped matching would make every one of
them pass on nothing. This is the number that says the list is real; raise it as the tree grows."""


def test_the_tree_is_read(files: list[str], imports: list[Import]) -> None:
    """A glob that matched nothing would make every rule below vacuous."""
    assert files and len(imports) >= MINIMUM_IMPORTS


PERMISSIONS = {library: homes for library, homes in QUARANTINE.items() if homes}
"""The rows that *allow* something; a row allowing nothing is a ban, and holds for an absent library."""


@pytest.mark.parametrize(("library", "homes"), PERMISSIONS.items(), ids=PERMISSIONS)
def test_every_quarantine_that_permits_something_permits_a_library_the_tree_imports(
    imports: list[Import], library: str, homes: tuple[str, ...]
) -> None:
    """A permission says "this is needed here". One nothing imports is a rule about nobody, and it
    outlives the code that earned it — which is how a quarantine drifts into decoration."""
    assert [one.file for one in imports if one.library == library] != [], f"{homes} import no {library}"


@pytest.mark.parametrize(("library", "homes"), QUARANTINE.items(), ids=QUARANTINE)
def test_every_home_a_quarantine_names_is_a_place_in_the_tree(
    files: list[str], library: str, homes: tuple[str, ...]
) -> None:
    """A home written ahead of the file it names reads as a permission and grants none; it also never fails."""
    missing = sorted(home for home in homes if not any(name.startswith(home) for name in files))

    assert missing == [], f"{library} is quarantined to paths that do not exist"


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


def test_every_package_declares_the_edges_it_may_use(files: list[str]) -> None:
    """A rule nobody wrote is a rule nobody breaks: an undeclared package would simply not be checked."""
    modules = {name.removesuffix(".py") for name in files if "/" not in name and name != "__init__.py"}

    undeclared = sorted((set(PACKAGES) | modules) - CAPABILITY_EDGES.keys())
    unwritten = sorted(CAPABILITY_EDGES.keys() - (set(PACKAGES) | modules))

    assert undeclared == [], "these may import anything until they are declared"
    assert unwritten == [], "these declare edges for something that is not there"


def test_every_edge_a_package_declares_points_at_a_package_that_exists(files: list[str]) -> None:
    """The same rule the quarantine is held to: a permission ahead of its target never fails and grants all."""
    modules = {name.removesuffix(".py") for name in files if "/" not in name and name != "__init__.py"}
    known = set(PACKAGES) | modules
    nowhere = sorted(
        f"{package} -> {target}" for package, allowed in CAPABILITY_EDGES.items() for target in allowed - known
    )

    assert nowhere == [], "these edges permit an import of something that is not in the tree"


@pytest.mark.parametrize(("package", "allowed"), sorted(CAPABILITY_EDGES.items()), ids=sorted(CAPABILITY_EDGES))
def test_every_edge_a_package_declares_is_one_it_travels(
    imports: list[Import], package: str, allowed: frozenset[str]
) -> None:
    """The rule the quarantine is already held to, applied to the other half of the table.

    An edge nothing travels is a permission about nobody, and it outlives whatever earned it — which
    is how a layering table drifts from a decision into decoration.
    """
    travelled = {one.target for one in imports if one.package == package and one.target is not None}

    assert sorted(allowed - travelled) == [], f"{package} declares edges it does not use"


@pytest.mark.parametrize(("package", "allowed"), sorted(CAPABILITY_EDGES.items()), ids=sorted(CAPABILITY_EDGES))
def test_a_capability_consumes_others_only_along_its_declared_edges(
    imports: list[Import], package: str, allowed: frozenset[str]
) -> None:
    undeclared = sorted(
        f"{one.file} imports {one.module}"
        for one in imports
        if one.package == package and one.target is not None and one.target not in {package, "core", "config"} | allowed
    )

    assert undeclared == []


def test_visualization_is_a_library_of_its_own(imports: list[Import], files: list[str]) -> None:
    """It is meant to be lifted into a package of its own, and an import of ours is what would stop that.

    Held over the whole directory rather than declared in prose: the cheapest way to reach a task's
    facts from a drawer is one import, and it would not look wrong in review.
    """
    assert any(name.startswith(VISUALIZATION) for name in files), "this rule has no subject"
    reaching_out = sorted(
        f"{one.file} imports {one.module}"
        for one in imports
        if one.file.startswith(VISUALIZATION)
        and one.library not in sys.stdlib_module_names
        and not one.module.startswith((*VISUALIZATION_MAY_IMPORT, "src.visualization"))
    )

    assert reaching_out == []


def test_only_build_modules_read_config(imports: list[Import]) -> None:
    """A capability receives ready objects; the declaration is read once, at the root or in its own build."""
    readers = sorted(
        one.file
        for one in imports
        if one.target == "config" and not one.file.startswith("config/") and Path(one.file).name not in CONFIG_READERS
    )

    assert readers == []


def test_the_edges_a_package_declares_never_lead_back_to_it() -> None:
    """Every other rule here reads one package's imports; a cycle is a property of two and passes them all.

    ``a`` may declare ``b`` and ``b`` may declare ``a``, each a legal row on its own, and together they
    make one unit out of two packages: neither can be read, tested, or lifted out without the other.
    Asked of the declarations rather than the imports, because a second edge is written here first —
    the import that needs it does not compile until it is.
    """
    reaches = {package: set(edges) for package, edges in CAPABILITY_EDGES.items()}
    for through, beyond in list(reaches.items()):
        for edges in reaches.values():
            if through in edges:
                edges.update(beyond)
    looping = sorted(package for package, edges in reaches.items() if package in edges)

    assert looping == [], "these packages can reach themselves"


def test_the_training_module_reads_capabilities_through_their_contracts_only(imports: list[Import]) -> None:
    if not any(one.file == TRAINING_MODULE for one in imports):
        pytest.skip(f"{TRAINING_MODULE} is not written yet; this rule has no subject to hold")
    reaching_in = sorted(
        one.module
        for one in imports
        if one.file == TRAINING_MODULE
        and one.target is not None
        and one.target != Path(TRAINING_MODULE).parts[0]
        and one.module not in TRAINING_MODULE_FACADES
    )

    assert reaching_in == []
