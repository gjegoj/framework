"""The export phase: every declared format, written from the weights in memory and proven."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from src.export.verification import ExportedArtifact, render_report, verify

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from torch import Tensor

    from src.export.deployable import DeployableModel
    from src.export.exporters import Exporter

log = logging.getLogger(__name__)

TRACING_BATCH = 2
"""The batch a graph is traced at: one risks collapsing squeeze-shaped operations."""

DEPLOYMENT_BATCH = 1
"""The single-sample shape deployment uses — the second shape verification runs."""


class ExampleRejected(ValueError):
    """The graph would not run on its example; the message names the shapes it was given.

    Its own class so the caller that shaped the example can say where the shape came
    from, which this module does not know.
    """


def ship(
    graph: DeployableModel,
    example: Callable[[int], tuple[Tensor, ...]],
    exporters: Sequence[Exporter],
    destination: Path,
) -> list[ExportedArtifact]:
    """Write every format from the weights in memory, and prove each one.

    The graph is moved to CPU because that is the portable place to trace from; a backend
    needing another device (a TensorRT engine) moves it itself. The run is over by the
    time this is called, so the move costs nothing. ``example`` shapes an input batch of
    the size asked for: tracing wants two rows, and verification also runs the single
    row deployment uses, so a graph that baked its batch is caught here.

    Raises:
        ExampleRejected: If the graph would not run on its example — before any file is written.
        RuntimeError: If any written artifact drifted outside its tolerance.
    """
    graph.eval()
    graph.cpu()
    tracing = example(TRACING_BATCH)
    _refuse_a_wrong_example(graph, tracing)
    examples = (tracing, example(DEPLOYMENT_BATCH))
    artifacts: list[ExportedArtifact] = []
    for exporter in exporters:
        path = exporter.export(graph, tracing, destination)
        parity = verify(exporter, path, graph, examples, atol=exporter.atol, rtol=exporter.rtol)
        log.info("Exported %s", path)
        artifacts.append(ExportedArtifact(path=path, parity=parity))
    render_report(artifacts)
    _refuse_drift(artifacts)
    return artifacts


def _refuse_a_wrong_example(graph: DeployableModel, example: tuple[Tensor, ...]) -> None:
    """One forward, negligible beside tracing the whole graph, so a wrong example fails here and by name."""
    try:
        with torch.no_grad():
            graph(*example)
    except Exception as error:
        shapes = ", ".join(
            f"{name} {tuple(tensor.shape)}" for name, tensor in zip(graph.input_names, example, strict=True)
        )
        raise ExampleRejected(f"The model rejected its export example ({shapes}).") from error


def _refuse_drift(artifacts: Sequence[ExportedArtifact]) -> None:
    drifted = [artifact for artifact in artifacts if not artifact.parity.within_tolerance]
    if drifted:
        names = ", ".join(f"{artifact.path.name} (abs {artifact.parity.max_abs:.2e})" for artifact in drifted)
        raise RuntimeError(f"Export verification failed for {names}; the written artifacts are not the model.")
