"""Proving a written artifact is still the model it came from."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from rich.table import Table

from src.console import console
from src.export.deployable import as_outputs

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from torch import Tensor

    from src.export.deployable import DeployableModel
    from src.export.exporters import Exporter

_RELATIVE_FLOOR = 1e-8
"""Keeps the relative error finite where the reference value is zero."""


@dataclass(frozen=True, slots=True)
class Parity:
    """How far a written artifact drifted from the model it was written from.

    ``per_output`` holds ``(absolute, relative)`` per output name — the worst
    seen across every example the artifact was run on. The two maxima are
    derived rather than stored, so a report cannot disagree with its own rows.
    """

    per_output: dict[str, tuple[float, float]]
    within_tolerance: bool

    @property
    def max_abs(self) -> float:
        return max((absolute for absolute, _ in self.per_output.values()), default=0.0)

    @property
    def max_rel(self) -> float:
        return max((relative for _, relative in self.per_output.values()), default=0.0)


@dataclass(frozen=True, slots=True)
class ExportedArtifact:
    """One written file and the proof it still is the model.

    No format field: the file name already says which format wrote it.
    """

    path: Path
    parity: Parity


def verify(
    exporter: Exporter,
    path: Path,
    model: DeployableModel,
    examples: Sequence[tuple[Tensor, ...]],
    *,
    atol: float,
    rtol: float,
) -> Parity:
    """Run the written artifact beside the model and report how far apart they are.

    Every example carries its own eager reference — outputs of different batch
    sizes have nothing to compare against each other — which is what lets a
    second example measure whether the graph baked the shape it was traced at.

    Parameters:
        exporter (Exporter): The backend that wrote ``path``; it also reads it back.
        path (Path): The written artifact.
        model (DeployableModel): The graph the artifact was written from.
        examples (Sequence[tuple[Tensor, ...]]): Inputs to run through both.
        atol (float): Absolute tolerance of the verdict.
        rtol (float): Relative tolerance of the verdict.
    """
    runnable = exporter.load(path)
    per_output: dict[str, tuple[float, float]] = {}
    within = True
    for example in examples:
        with torch.no_grad():
            reference = as_outputs(model(*example))
        written = runnable(example)
        if len(written) != len(reference):
            raise ValueError(
                f"{path.name} returns {len(written)} output(s) where the model returns "
                f"{len(reference)} ({', '.join(model.output_names)})."
            )
        for name, expected, actual in zip(model.output_names, reference, written, strict=True):
            absolute, relative, agrees = _error(expected, actual, atol=atol, rtol=rtol)
            worst_absolute, worst_relative = per_output.get(name, (0.0, 0.0))
            per_output[name] = (max(worst_absolute, absolute), max(worst_relative, relative))
            within = within and agrees
    return Parity(per_output=per_output, within_tolerance=within)


def _error(expected: Tensor, actual: Tensor, *, atol: float, rtol: float) -> tuple[float, float, bool]:
    """One output's absolute and relative drift, and whether it is tolerated.

    A shape mismatch is infinite drift rather than a comparison: broadcasting two
    differently shaped outputs would invent a number for a graph that is simply
    wrong.
    """
    if expected.shape != actual.shape:
        return float("inf"), float("inf"), False
    reference = expected.detach().cpu().float()
    written = actual.detach().cpu().float()
    difference = (reference - written).abs()
    if not difference.numel():
        return 0.0, 0.0, True
    relative = difference / (reference.abs() + _RELATIVE_FLOOR)
    # The allclose form: the absolute term dominates near zero, where a pure
    # relative error explodes on a difference nobody would call a drift.
    agrees = bool((difference <= atol + rtol * reference.abs()).all())
    return float(difference.max()), float(relative.max()), agrees


def render_report(artifacts: Sequence[ExportedArtifact]) -> None:
    """Draw what verification found: one row per output, one verdict per artifact.

    Separate from ``verify`` on purpose: a verdict that exists only as printed text
    cannot be asserted, and a report that can only be printed cannot be sent anywhere
    else.
    """
    table = Table(title="Export verification")
    table.add_column("Artifact")
    table.add_column("Output")
    table.add_column("Absolute")
    table.add_column("Relative")
    for artifact in artifacts:
        for name, (absolute, relative) in artifact.parity.per_output.items():
            table.add_row(artifact.path.name, name, f"{absolute:.2e}", f"{relative:.2e}")
        verdict = "[green]within tolerance[/]" if artifact.parity.within_tolerance else "[red]EXCEEDS tolerance[/]"
        table.add_row(artifact.path.name, "verdict", verdict, "")
    console().print(table)
