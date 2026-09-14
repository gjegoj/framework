"""Shipping what a run trained: the graph a format writes, and the formats it can be written in.

Importing this package is what makes the names an `export` declaration may write resolvable, as in
every package here: each backend registers while its own module runs.
"""

from __future__ import annotations

from src.export.backends import (
    NcnnExporter,
    OnnxExporter,
    Pt2Exporter,
    TensorRtExporter,
    TorchScriptExporter,
)
from src.export.base import WRITTEN_FROM, Exporter, Runnable
from src.export.deployable import DeployableModel, as_outputs, example_inputs
from src.export.manifest import Manifest, ship
from src.export.verification import Parity, verify

__all__ = [
    "WRITTEN_FROM",
    "DeployableModel",
    "Exporter",
    "Manifest",
    "NcnnExporter",
    "OnnxExporter",
    "Parity",
    "Pt2Exporter",
    "Runnable",
    "TensorRtExporter",
    "TorchScriptExporter",
    "as_outputs",
    "example_inputs",
    "ship",
    "verify",
]
