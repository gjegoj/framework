"""The export capability: a trained model becomes a file a serving stack loads."""

from __future__ import annotations

from src.export.backends import OnnxExporter, TensorRtExporter, TorchScriptExporter
from src.export.deployable import DeployableModel, as_outputs
from src.export.exporters import Exporter, Runnable
from src.export.registry import exporter_registry
from src.export.verification import ExportedArtifact, Parity, render_report, verify

__all__ = [
    "DeployableModel",
    "ExportedArtifact",
    "Exporter",
    "OnnxExporter",
    "Parity",
    "Runnable",
    "TensorRtExporter",
    "TorchScriptExporter",
    "as_outputs",
    "exporter_registry",
    "render_report",
    "verify",
]
