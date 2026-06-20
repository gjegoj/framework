"""Model export pipeline: plan → emissions → registered exporters.

``export_model`` is a thin planner: it turns the export config into a declarative
list of ``_Emission``s (what to export — a traceable wrapper plus its naming), then
emits each through every configured format. ``_emit`` owns the how — trace, export,
verify, log, gate — for one wrapper across all targets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from src.config.schema import ExperimentConfig
from src.core.entities import Task
from src.core.ports import Head
from src.export.entities import ExportArtifact, ExportArtifactKind, ExportRequest
from src.export.registry import exporters
from src.export.spec import ExportPlan, build_export_plan
from src.export.tracing import as_output_tuple
from src.export.verify import render_report, verify_artifact
from src.export.wrapper import BackboneExportModel, CombinedExportModel, HeadExportModel
from src.models.assembly import CompositeModel

log = logging.getLogger(__name__)

# Verification knobs live on the typed target (read straight into verify_artifact); they
# must not leak into the exporter-facing options dict carried by ExportRequest.
_VERIFY_FIELDS = frozenset({"verify_outputs", "atol", "rtol"})


def resolve_export_io_names(semantic_names: list[str], *, prefix: str, generic: bool) -> list[str]:
    """Map semantic tensor names to deployment-facing I/O names.

    Parameters:
        semantic_names (list[str]): Internal names (task names, stream keys, …).
        prefix (str): ``input`` or ``output``.
        generic (bool): When true, use ``prefix`` / ``prefix_N``; else keep ``semantic_names``.

    Returns:
        list[str]: Names passed to the exporter.
    """
    if not generic:
        return list(semantic_names)
    if len(semantic_names) == 1:
        return [prefix]
    return [f"{prefix}_{index}" for index in range(len(semantic_names))]


@dataclass(frozen=True)
class _Emission:
    """One artifact to export: a traceable wrapper plus its naming and identity.

    Parameters:
        wrapper (nn.Module): Tensor-in / tensor-out wrapper to trace.
        example_inputs (tuple[Tensor, ...]): Dummy inputs for tracing / reference outputs.
        basename (Path): Output path without suffix (each exporter appends its own).
        input_semantic (list[str]): Semantic input names, before generic-io mapping.
        output_semantic (list[str]): Semantic output names, before generic-io mapping.
        kind (ExportArtifactKind): ``combined`` / ``backbone`` / ``head``.
        task_name (str | None): Task name for ``head`` emissions; ``None`` otherwise.
    """

    wrapper: nn.Module
    example_inputs: tuple[Tensor, ...]
    basename: Path
    input_semantic: list[str]
    output_semantic: list[str]
    kind: ExportArtifactKind
    task_name: str | None = None

    @property
    def display_label(self) -> str:
        """Human label for logs: ``head 'name'`` for heads, else the kind (``combined``/``backbone``)."""
        return f"head '{self.task_name}'" if self.task_name is not None else self.kind


def _prepare_module(module: nn.Module) -> nn.Module:
    module.eval()
    return module.cpu()


def _reference_outputs(module: nn.Module, example_inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    """Run the prepared source wrapper once; its outputs are the parity ground truth."""
    with torch.no_grad():
        return as_output_tuple(module(*example_inputs))


def _emit(emission: _Emission, config: ExperimentConfig) -> list[ExportArtifact]:
    """Export one wrapper across every configured format: trace, export, verify, log, gate."""
    module = _prepare_module(emission.wrapper)
    reference = _reference_outputs(module, emission.example_inputs)
    generic = config.export.generic_io_names
    input_names = resolve_export_io_names(emission.input_semantic, prefix="input", generic=generic)
    output_names = resolve_export_io_names(emission.output_semantic, prefix="output", generic=generic)

    artifacts: list[ExportArtifact] = []
    for target in config.export.targets:
        format_name = target.format
        # Exporter knobs only: verification tolerances are read off the typed target below.
        options: dict[str, object] = target.model_dump(exclude={"format", *_VERIFY_FIELDS})
        exporter = exporters.create(format_name)
        path = emission.basename.with_suffix(exporter.extension)
        request = ExportRequest(
            module=module,
            example_inputs=emission.example_inputs,
            path=path,
            input_names=input_names,
            output_names=output_names,
            options=options,
        )
        exporter.export(request)
        report = verify_artifact(
            exporter,
            request,
            reference,
            atol=target.atol,
            rtol=target.rtol,
            run_parity=target.verify_outputs,
        )
        artifact = ExportArtifact(
            path=path, format=format_name, kind=emission.kind, name=emission.task_name, report=report
        )
        artifacts.append(artifact)
        log.info("Exported %s %s → %s", emission.display_label, format_name, path)
        _report_and_gate(artifact)
    return artifacts


def _report_and_gate(artifact: ExportArtifact) -> None:
    """Render the verification report (if any), then raise if it did not pass."""
    report = artifact.report
    if report is None:
        return
    render_report(artifact)
    if not report.ok:
        raise RuntimeError(f"Export verification failed for {artifact.path}: {report.failure_summary}")


def _plan_emissions(
    model: CompositeModel,
    plan: ExportPlan,
    task_by_name: dict[str, Task],
    config: ExperimentConfig,
    output_dir: Path,
) -> list[_Emission]:
    """Turn the export config into the ordered list of wrappers to emit."""
    image_inputs = (plan.dummy_image.clone(),)
    emissions: list[_Emission] = []

    if config.export.combined:
        activations = {name: task_by_name[name].activation for name in plan.task_names}
        emissions.append(
            _Emission(
                wrapper=CombinedExportModel(model, plan.task_names, activations, input_key=plan.input_key),
                example_inputs=image_inputs,
                basename=output_dir / "model_combined",
                input_semantic=[plan.input_key],
                output_semantic=list(plan.task_names),
                kind="combined",
            )
        )

    if config.export.split_components:
        emissions.append(
            _Emission(
                wrapper=BackboneExportModel(model.backbone, plan.stream_keys, input_key=plan.input_key),
                example_inputs=image_inputs,
                basename=output_dir / "backbone",
                input_semantic=[plan.input_key],
                output_semantic=list(plan.stream_keys),
                kind="backbone",
            )
        )
        for task_name in plan.task_names:
            spec = plan.head_specs[task_name]
            emissions.append(
                _Emission(
                    wrapper=HeadExportModel(
                        cast(Head, model.heads[task_name]),
                        activation=task_by_name[task_name].activation,
                    ),
                    example_inputs=(spec.dummy_features.clone(),),
                    basename=output_dir / f"head_{task_name}",
                    input_semantic=[spec.feature_key],
                    output_semantic=[task_name],
                    kind="head",
                    task_name=task_name,
                )
            )
    return emissions


def export_model(
    model: CompositeModel,
    tasks: list[Task],
    config: ExperimentConfig,
    output_dir: Path,
) -> list[ExportArtifact]:
    """Export ``model`` according to ``config.export``.

    Parameters:
        model (CompositeModel): Assembled model with final weights for deployment.
        tasks (list[Task]): Active tasks.
        config (ExperimentConfig): Validated experiment config.
        output_dir (Path): Directory for artifact files.

    Returns:
        list[ExportArtifact]: Written files.
    """
    if not config.export.targets:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_export_plan(model, tasks, config)
    _prepare_module(model)
    task_by_name = {task.name: task for task in tasks}

    artifacts: list[ExportArtifact] = []
    for emission in _plan_emissions(model, plan, task_by_name, config, output_dir):
        artifacts.extend(_emit(emission, config))
    return artifacts
