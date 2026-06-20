"""Generic export verification: numerical parity + report assembly and rendering.

Format-agnostic use-case. The only per-format pieces are the exporter's optional
``load()`` (a callable runner) and ``validate()`` (static checks → dict); this
module composes them into an :class:`ExportReport`. The source model's reference
outputs are computed once by the pipeline and injected, so adding a format costs
nothing here (OCP).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from rich import print as rprint
from rich.table import Table
from torch import Tensor

from src.export.entities import ExportReport, ParityResult

if TYPE_CHECKING:
    from src.export.entities import ExportArtifact, ExportRequest
    from src.export.ports import ModelExporter

_REL_EPS = 1e-8


def compute_parity(
    reference: tuple[Tensor, ...],
    actual: tuple[Tensor, ...],
    names: list[str],
    *,
    atol: float,
    rtol: float,
) -> ParityResult:
    """Compute max abs/rel output error between source and exported outputs.

    Parameters:
        reference (tuple[Tensor, ...]): Source model outputs (ground truth).
        actual (tuple[Tensor, ...]): Exported artifact outputs (same order/count).
        names (list[str]): Output names, used as ``per_output`` keys.
        atol (float): Absolute tolerance for the verdict.
        rtol (float): Relative tolerance for the verdict.

    Returns:
        ParityResult: Aggregated and per-output errors plus the tolerance verdict.
    """
    per_output: dict[str, tuple[float, float]] = {}
    max_abs = 0.0
    max_rel = 0.0
    within = True
    for name, reference_tensor, actual_tensor in zip(names, reference, actual, strict=True):
        reference_float = reference_tensor.detach().cpu().float()
        actual_float = actual_tensor.detach().cpu().float()
        abs_err = (reference_float - actual_float).abs()
        rel_err = abs_err / (reference_float.abs() + _REL_EPS)
        abs_value = float(abs_err.max()) if abs_err.numel() else 0.0
        rel_value = float(rel_err.max()) if rel_err.numel() else 0.0
        per_output[name] = (abs_value, rel_value)
        max_abs = max(max_abs, abs_value)
        max_rel = max(max_rel, rel_value)
        # Combined tolerance (numpy.allclose form): |a - b| <= atol + rtol * |b|.
        # The atol term dominates near zero, where a pure relative error explodes.
        if abs_err.numel():
            within = within and bool((abs_err <= atol + rtol * reference_float.abs()).all())
    return ParityResult(max_abs_error=max_abs, max_rel_error=max_rel, within_tolerance=within, per_output=per_output)


def verify_artifact(
    exporter: ModelExporter,
    request: ExportRequest,
    reference: tuple[Tensor, ...],
    *,
    atol: float,
    rtol: float,
    run_parity: bool,
) -> ExportReport | None:
    """Verify one written artifact: static checks + optional numerical parity.

    Builds the full report without raising (the caller decides policy). Returns
    ``None`` when the backend contributes neither checks nor a runnable.

    Parameters:
        exporter (ModelExporter): The backend that wrote ``request.path``.
        request (ExportRequest): The export invocation (path, inputs, names, options).
        reference (tuple[Tensor, ...]): Source model outputs to compare against.
        atol (float): Absolute parity tolerance.
        rtol (float): Relative parity tolerance.
        run_parity (bool): Whether to run the numerical parity check.

    Returns:
        ExportReport | None: The verification report, or ``None`` if empty.
    """
    checks = exporter.validate(request)
    parity: ParityResult | None = None
    if run_parity:
        runnable = exporter.load(request.path)
        if runnable is not None:
            feeds = dict(zip(request.input_names, request.example_inputs, strict=True))
            with torch.no_grad():
                actual = runnable(feeds)
            parity = compute_parity(reference, actual, request.output_names, atol=atol, rtol=rtol)
    if not checks and parity is None:
        return None
    return ExportReport(checks=checks, parity=parity)


def render_report(artifact: ExportArtifact) -> None:
    """Print a rich table summarizing one artifact's verification report."""
    report = artifact.report
    if report is None:
        return
    subtitle = f"{artifact.format} · {artifact.kind}" + (f" · {artifact.name}" if artifact.name else "")
    table = Table(title=f"Export verification — {subtitle}")
    table.add_column("Check")
    table.add_column("Result")
    for name, detail in report.checks.items():
        table.add_row(name, "[green]PASS[/]" if not detail else f"[red]FAIL[/] {detail}")
    if report.parity is not None:
        parity = report.parity
        for name, (abs_error, rel_error) in parity.per_output.items():
            table.add_row(name, f"abs={abs_error:.2e}  rel={rel_error:.2e}")
        verdict = "[green]within tol[/]" if parity.within_tolerance else "[red]EXCEEDS tol[/]"
        table.add_row("parity (max)", f"abs={parity.max_abs_error:.2e}  rel={parity.max_rel_error:.2e}  {verdict}")
    rprint(table)
