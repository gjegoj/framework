"""The export phase: what to write, from what, and where."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from src.assembly.instantiate import instantiate
from src.export import DeployableModel, ExportedArtifact, exporter_registry, render_report, verify

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from src.config import ExperimentConfig
    from src.core.ports import Model
    from src.export import Exporter

log = logging.getLogger(__name__)

TRACING_BATCH = 2
"""The batch a graph is traced at: one risks collapsing squeeze-shaped operations."""

DEPLOYMENT_BATCH = 1
"""The single-sample shape deployment uses — the second shape verification runs."""


def build_exporters(config: ExperimentConfig) -> list[Exporter]:
    """The declared deployment formats, refusing two that would write one file."""
    declared = config.export or []
    counted = Counter(entry.target or entry.name for entry in declared)
    repeated = sorted(str(reference) for reference, count in counted.items() if count > 1)
    if repeated:
        raise ValueError(
            f"The export section declares {', '.join(repeated)} more than once, and a second target of "
            "one format would overwrite the first. Declare each format once."
        )
    return [instantiate(entry, exporter_registry) for entry in declared]


def example_inputs(config: ExperimentConfig, batch_size: int) -> tuple[Tensor, ...]:
    """One dummy tensor per declared model input, shaped the way the run feeds them.

    Not a guess: ``image_size`` and ``mean`` are the very fields the transform
    pipeline hands to ``Resize`` and ``Normalize``, so this is the shape the model
    receives by construction of the config contract — and export therefore needs
    no dataset, which is what lets it run from a checkpoint anywhere.
    """
    channels = len(config.mean)
    height, width = config.image_size
    return tuple(torch.randn(batch_size, channels, height, width) for _ in config.data.inputs)


def export_model(model: Model, config: ExperimentConfig, exporters: Sequence[Exporter]) -> list[ExportedArtifact]:
    """Write every declared format from the weights in memory, and prove each one.

    The graph is moved to CPU because that is the portable place to trace from;
    a backend needing another device (a TensorRT engine) moves it itself. The run
    is over by the time this is called, so the move costs nothing.

    Raises:
        ValueError: If the example shape is not what the model takes.
        RuntimeError: If any written artifact drifted outside its tolerance.
    """
    graph = DeployableModel(model, list(config.data.inputs), list(config.tasks))
    graph.eval()
    graph.cpu()
    tracing = example_inputs(config, TRACING_BATCH)
    _prove_the_example_fits(graph, tracing, config)
    examples = (tracing, example_inputs(config, DEPLOYMENT_BATCH))
    destination = Path(config.run.directory or ".") / "export" / "model"

    artifacts: list[ExportedArtifact] = []
    for exporter in exporters:
        path = exporter.export(graph, tracing, destination)
        parity = verify(exporter, path, graph, examples, atol=exporter.atol, rtol=exporter.rtol)
        log.info("Exported %s", path)
        artifacts.append(ExportedArtifact(path=path, parity=parity))
    render_report(artifacts)

    drifted = [artifact for artifact in artifacts if not artifact.parity.within_tolerance]
    if drifted:
        names = ", ".join(f"{artifact.path.name} (abs {artifact.parity.max_abs:.2e})" for artifact in drifted)
        raise RuntimeError(f"Export verification failed for {names}; the written artifacts are not the model.")
    return artifacts


def _prove_the_example_fits(graph: DeployableModel, example: tuple[Tensor, ...], config: ExperimentConfig) -> None:
    """Run the graph once before any exporter does, so a wrong example fails here and by name.

    One forward, negligible beside tracing the whole graph, and it turns a raw
    torch shape error into the two config fields the user can actually change.
    """
    try:
        with torch.no_grad():
            graph(*example)
    except Exception as error:
        shapes = ", ".join(
            f"{name} {tuple(tensor.shape)}" for name, tensor in zip(graph.input_names, example, strict=True)
        )
        raise ValueError(
            f"The model rejected its export example ({shapes}). The shape comes from 'image_size' "
            f"{tuple(config.image_size)} and the {len(config.mean)} normalisation channel(s) of 'mean'; "
            "an input that is not an image of that shape cannot be exported yet."
        ) from error
