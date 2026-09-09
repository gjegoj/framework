"""The dependency rule as a test: each third-party stack lives in the package that quarantines it.

``concepts.md`` says arrows point down only and names where each library is confined —
Lightning in ``training/``, pydantic in ``config/``, Hydra's composition in ``cli.py``,
albumentations behind a seam. A rule stated in prose holds until the first convenient
import; this reads every import in ``src/`` and names the file that broke it.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC = Path(__file__).parents[2] / "src"

QUARANTINE: dict[str, tuple[str, ...]] = {
    "lightning": ("training/", "callbacks/", "loggers/", "build.py"),
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
    # smp's losses serve the segmentation criteria: the one leak the audit chose to note rather than seal.
    "segmentation_models_pytorch": ("models/backbones/", "losses/segmentation.py"),
    "torchmetrics": ("metrics/",),
    "clearml": ("loggers/",),
    "plotly": ("loggers/",),
    "cv2": ("data/",),
    "pandas": ("data/",),
    "sklearn": ("data/",),
    "skmultilearn": ("data/",),
    "onnx": ("export/",),
    "onnxruntime": ("export/",),
    "onnxscript": ("export/",),
    "onnxsim": ("export/",),
    "tensorrt": ("export/",),
    # Presentation. Where it is today, by measurement; narrowing it is the owner's open item.
    "rich": ("console.py", "progress.py", "cli.py", "callbacks/", "export/verification.py"),
}
"""Library → the paths under ``src/`` allowed to import it, by package prefix so a new module in the package needs no edit here."""

CORE_MAY_IMPORT = ("torch", "src.core")
"""Besides the standard library: ``core/`` is the port every package implements, so it knows no package."""

MODULE_READS_ONLY_PORTS = {"src.metrics.ports", "src.loggers.report"}
"""``training/module.py`` asks a metric set and a logger what their ports promise, never how a package builds them."""

CAPABILITY_EDGES: dict[str, frozenset[str]] = {
    "callbacks": frozenset(
        {"console", "data", "loggers", "models", "tasks", "training", "transforms", "visualization"}
    ),
    "data": frozenset({"progress", "tasks"}),
    "export": frozenset({"console"}),
    "loggers": frozenset({"data", "metrics"}),
    "metrics": frozenset({"tasks"}),
    "tasks": frozenset({"losses", "models", "visualization"}),
    "training": frozenset({"data", "loggers", "metrics", "tasks"}),
    "transforms": frozenset({"tasks"}),
}
"""Which capability may import which, besides ``core`` and ``config`` (leaves every package may read).

Measured on 2026-09-09 and pinned: a capability consumes another only through what it publishes,
the graph is a DAG, and a new edge is a decision recorded here, not a drift found later.
"""

VISUALIZATION_MAY_IMPORT = ("src.visualization", "src.core")
"""ADR-0003 keeps ``visualization/`` a leaf: a page draws what it is handed, so nothing it
imports may pull a task, a model or a metric in behind it."""


def imports_of(path: Path) -> list[str]:
    """Every module a file imports, nested imports included — a lazy import is still a dependency."""
    names: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
    return names


def every_import() -> list[tuple[str, str]]:
    """``(file, module)`` pairs over the whole tree, files relative to ``src/``."""
    return [(path.relative_to(SRC).as_posix(), name) for path in sorted(SRC.rglob("*.py")) for name in imports_of(path)]


def test_the_tree_is_read() -> None:
    """A glob that matched nothing would make the two tests below vacuous."""
    assert len(every_import()) > 100


def test_every_quarantined_library_is_imported_only_inside_its_package() -> None:
    strays = sorted(
        f"{file} imports {module}"
        for file, module in every_import()
        if (library := module.split(".")[0]) in QUARANTINE and not file.startswith(QUARANTINE[library])
    )

    assert strays == []


def test_core_imports_torch_the_standard_library_and_itself_only() -> None:
    reaching_out = sorted(
        f"{file} imports {module}"
        for file, module in every_import()
        if file.startswith("core/")
        and module.split(".")[0] not in sys.stdlib_module_names
        and not module.startswith(CORE_MAY_IMPORT)
    )

    assert reaching_out == []


def test_the_training_module_reads_the_capabilities_through_their_ports_only() -> None:
    capabilities = {
        module
        for file, module in every_import()
        if file == "training/module.py"
        and module.startswith(("src.metrics", "src.loggers", "src.data", "src.models", "src.losses"))
    }

    assert capabilities <= MODULE_READS_ONLY_PORTS, capabilities


def test_the_visualization_package_reaches_into_no_other_capability() -> None:
    """A drawer is handed the two facts it labels with; it never reaches back for the task that has them."""
    reaching_out = sorted(
        f"{file} imports {module}"
        for file, module in every_import()
        if file.startswith("visualization/")
        and module.startswith("src.")
        and not module.startswith(VISUALIZATION_MAY_IMPORT)
    )

    assert reaching_out == []


def test_capabilities_consume_each_other_only_along_the_declared_edges() -> None:
    """The root wires everything; between capabilities the arrows are few, named, and pinned."""
    undeclared = sorted(
        f"{file} imports {module}"
        for file, module in every_import()
        if module.startswith("src.")
        and (source := file.split("/")[0].removesuffix(".py")) in CAPABILITY_EDGES
        and (target := module.split(".")[1]) not in {source, "core", "config"}
        and target not in CAPABILITY_EDGES[source]
    )

    assert undeclared == []
