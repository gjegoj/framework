"""Export domain objects — artifacts produced by the export pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ExportArtifactKind = Literal["combined", "backbone", "head"]
ExportFormat = Literal["onnx", "torchscript"]


@dataclass(frozen=True)
class ExportArtifact:
    """One exported model file on disk.

    Parameters:
        path (Path): Output file path.
        format (ExportFormat): Exporter registry key.
        kind (ExportArtifactKind): Logical role — ``combined``, ``backbone``, or ``head``.
        name (str | None): Task name for ``kind='head'``; ``None`` otherwise.
    """

    path: Path
    format: ExportFormat
    kind: ExportArtifactKind
    name: str | None = None
