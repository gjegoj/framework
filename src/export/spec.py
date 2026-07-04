"""Export planning: dummy inputs, stream keys, Phase-1 guards."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.config.schema import ExperimentConfig
from src.core.entities import Task
from src.core.taxonomy import EXPORTABLE_TOPOLOGIES
from src.data.loaders import input_aliases
from src.models.assembly import CompositeModel


@dataclass(frozen=True)
class HeadExportSpec:
    """Dummy feature tensor shape for exporting one head in isolation.

    Parameters:
        task_name (str): Task / head name.
        feature_key (str): Backbone stream this head consumes.
        dummy_features (Tensor): Example input for ``HeadExportModel.forward``.
    """

    task_name: str
    feature_key: str
    dummy_features: Tensor


@dataclass(frozen=True)
class ExportPlan:
    """Everything an export run needs besides weights and format choice.

    Parameters:
        input_key (str): Model input alias (e.g. ``image``).
        input_channels (int): RGB channel count for the dummy image.
        image_size (tuple[int, int]): ``(H, W)`` from experiment config.
        task_names (tuple[str, ...]): Tasks in combined-output order.
        stream_keys (tuple[str, ...]): Unique backbone streams referenced by tasks.
        head_specs (dict[str, HeadExportSpec]): Per-task head export metadata.
        dummy_image (Tensor): Example image tensor used for combined/backbone tracing.
    """

    input_key: str
    input_channels: int
    image_size: tuple[int, int]
    task_names: tuple[str, ...]
    stream_keys: tuple[str, ...]
    head_specs: dict[str, HeadExportSpec]
    dummy_image: Tensor


def resolve_export_input_key(config: ExperimentConfig) -> str:
    """Resolve the single image input alias used for export tracing.

    Parameters:
        config (ExperimentConfig): Validated experiment config.

    Returns:
        str: Input alias (e.g. ``image``).

    Raises:
        ValueError: If multiple inputs are configured without ``export.input_key``.
    """
    if config.export.input_key is not None:
        return config.export.input_key
    aliases = input_aliases(config.data.inputs)
    if len(aliases) != 1:
        raise ValueError(
            f"export.input_key is required when data.inputs defines multiple aliases {aliases}. "
            "Set export.input_key to the image stream to trace."
        )
    return aliases[0]


def guard_exportable_topologies(tasks: list[Task]) -> None:
    """Raise if any task's topology is not exportable in Phase 1 (pure check).

    Needs only ``tasks`` + ``EXPORTABLE_TOPOLOGIES`` — no model, weights, or
    dummy forward — so it can run before training as a fail-fast guard.

    Parameters:
        tasks (list[Task]): Active tasks.

    Raises:
        ValueError: If a task topology is not in ``EXPORTABLE_TOPOLOGIES``.
    """
    for task in tasks:
        if task.topology not in EXPORTABLE_TOPOLOGIES:
            raise ValueError(
                f"Export does not yet support task '{task.name}' with topology "
                f"{task.topology}. Phase 1 supports "
                f"{sorted(topology.value for topology in EXPORTABLE_TOPOLOGIES)} only."
            )


def build_export_plan(model: CompositeModel, tasks: list[Task], config: ExperimentConfig) -> ExportPlan:
    """Build export metadata and validate Phase-1 (single-image) constraints.

    Parameters:
        model (CompositeModel): Assembled model (weights already loaded).
        tasks (list[Task]): Active tasks in declaration order.
        config (ExperimentConfig): Validated experiment config.

    Returns:
        ExportPlan: Dummy shapes and naming for exporters.

    Raises:
        ValueError: If any task topology is not exportable in Phase 1 (multiview / multistream).
    """
    guard_exportable_topologies(tasks)

    input_key = resolve_export_input_key(config)
    task_names = tuple(task.name for task in tasks)
    stream_keys = tuple(dict.fromkeys(task.head_spec.feature_key for task in tasks))

    channels = len(config.mean)
    height, width = config.image_size
    dummy_image = torch.randn(1, channels, height, width)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            bundle = model.backbone({input_key: dummy_image})
    finally:
        if was_training:
            model.train()

    head_specs: dict[str, HeadExportSpec] = {}
    for task in tasks:
        feature_key = task.head_spec.feature_key
        head_specs[task.name] = HeadExportSpec(
            task_name=task.name,
            feature_key=feature_key,
            dummy_features=bundle[feature_key].detach(),
        )

    return ExportPlan(
        input_key=input_key,
        input_channels=channels,
        image_size=config.image_size,
        task_names=task_names,
        stream_keys=stream_keys,
        head_specs=head_specs,
        dummy_image=dummy_image.detach(),
    )
