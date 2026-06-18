"""Export domain objects — artifacts and verification reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ExportArtifactKind = Literal["combined", "backbone", "head"]
ExportFormat = Literal["onnx", "torchscript", "tensorrt"]


@dataclass(frozen=True)
class ParityResult:
    """Numerical agreement between the exported artifact and the source model.

    Parameters:
        max_abs_error (float): Largest absolute output difference across all outputs.
        max_rel_error (float): Largest relative output difference across all outputs.
        within_tolerance (bool): Whether both errors are within the configured atol/rtol.
        per_output (dict[str, tuple[float, float]]): ``output_name -> (abs, rel)`` errors.
    """

    max_abs_error: float
    max_rel_error: float
    within_tolerance: bool
    per_output: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class ExportReport:
    """Verification outcome for one exported artifact.

    Parameters:
        checks (dict[str, str]): Static checks by name; value is ``""`` when the
            check passed, or the failure detail when it failed.
        parity (ParityResult | None): Numerical parity result (None when not run).
    """

    checks: dict[str, str] = field(default_factory=dict)
    parity: ParityResult | None = None

    @property
    def ok(self) -> bool:
        """True when no static check failed and parity (if run) is within tolerance."""
        return all(not detail for detail in self.checks.values()) and (
            self.parity is None or self.parity.within_tolerance
        )

    @property
    def failure_summary(self) -> str:
        """One-line description of what failed (empty when ``ok``); for exception messages."""
        parts = [name for name, detail in self.checks.items() if detail]
        if self.parity is not None and not self.parity.within_tolerance:
            parts.append(f"parity(abs={self.parity.max_abs_error:.2e}, rel={self.parity.max_rel_error:.2e})")
        return ", ".join(parts)


@dataclass(frozen=True)
class ExportArtifact:
    """One exported model file on disk.

    Parameters:
        path (Path): Output file path.
        format (ExportFormat): Exporter registry key.
        kind (ExportArtifactKind): Logical role — ``combined``, ``backbone``, or ``head``.
        name (str | None): Task name for ``kind='head'``; ``None`` otherwise.
        report (ExportReport | None): Verification report; ``None`` when verification produced nothing.
    """

    path: Path
    format: ExportFormat
    kind: ExportArtifactKind
    name: str | None = None
    report: ExportReport | None = None
