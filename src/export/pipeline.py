"""Model export pipeline: plan → wrappers → registered exporters."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from src.config.schema import ExperimentConfig
from src.core.entities import ExportRequest, Task
from src.core.ports import Head
from src.export.entities import ExportArtifact, ExportArtifactKind
from src.export.registry import exporters
from src.export.spec import build_export_plan
from src.export.verify import render_report, verify_artifact
from src.export.wrapper import BackboneExportModel, CombinedExportModel, HeadExportModel
from src.models.assembly import CompositeModel

log = logging.getLogger(__name__)


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


def _prepare_module(module: nn.Module) -> nn.Module:
    module.eval()
    return module.cpu()


def _reference_outputs(module: nn.Module, args: tuple) -> tuple[Tensor, ...]:
    """Run the prepared source wrapper once; its outputs are the parity ground truth."""
    with torch.no_grad():
        raw = module(*args) if len(args) > 1 else module(args[0])
    return raw if isinstance(raw, tuple) else (raw,)


def _emit(
    wrapper: nn.Module,
    args: tuple,
    *,
    basename: Path,
    input_names: list[str],
    output_names: list[str],
    kind: ExportArtifactKind,
    config: ExperimentConfig,
    task_name: str | None = None,
) -> list[ExportArtifact]:
    module = _prepare_module(wrapper)
    reference = _reference_outputs(module, args)
    artifacts: list[ExportArtifact] = []
    for target in config.export.targets:
        format_name = target.format
        # Exporter-level knobs travel via the neutral options dict; verification
        # knobs (atol/rtol/verify_outputs) are read off the typed target directly.
        options: dict[str, object] = target.model_dump(exclude={"format"})
        exporter = exporters.create(format_name)
        path = basename.with_suffix(exporter.extension)
        request = ExportRequest(
            module=module,
            example_inputs=args,
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
        artifact = ExportArtifact(path=path, format=format_name, kind=kind, name=task_name, report=report)
        artifacts.append(artifact)
        _log_export(kind, format_name, path, task_name)
        _report_and_gate(artifact)
    return artifacts


def _log_export(kind: ExportArtifactKind, format_name: str, path: Path, task_name: str | None) -> None:
    """Log the one-line 'exported X → path' message for an artifact."""
    if task_name is not None:
        log.info("Exported head '%s' %s → %s", task_name, format_name, path)
    elif kind == "backbone":
        log.info("Exported backbone %s → %s", format_name, path)
    else:
        log.info("Exported combined %s → %s", format_name, path)


def _report_and_gate(artifact: ExportArtifact) -> None:
    """Render the verification report (if any), then raise if it did not pass."""
    report = artifact.report
    if report is None:
        return
    render_report(artifact)
    if not report.ok:
        raise RuntimeError(f"Export verification failed for {artifact.path}: {report.failure_summary}")


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
    model = model.cpu()
    model.eval()
    task_by_name = {task.name: task for task in tasks}
    artifacts: list[ExportArtifact] = []
    generic_io = config.export.generic_io_names
    image_arg = (plan.dummy_image.clone(),)

    if config.export.combined:
        activations = {name: task_by_name[name].activation for name in plan.task_names}
        wrapper = CombinedExportModel(model, plan.task_names, activations, input_key=plan.input_key)
        artifacts.extend(
            _emit(
                wrapper,
                image_arg,
                basename=output_dir / "model_combined",
                input_names=resolve_export_io_names([plan.input_key], prefix="input", generic=generic_io),
                output_names=resolve_export_io_names(list(plan.task_names), prefix="output", generic=generic_io),
                kind="combined",
                config=config,
            )
        )

    if config.export.split_components:
        backbone_wrapper = BackboneExportModel(model.backbone, plan.stream_keys, input_key=plan.input_key)
        artifacts.extend(
            _emit(
                backbone_wrapper,
                image_arg,
                basename=output_dir / "backbone",
                input_names=resolve_export_io_names([plan.input_key], prefix="input", generic=generic_io),
                output_names=resolve_export_io_names(list(plan.stream_keys), prefix="output", generic=generic_io),
                kind="backbone",
                config=config,
            )
        )

        for task_name in plan.task_names:
            head_wrapper = HeadExportModel(
                cast(Head, model.heads[task_name]),
                activation=task_by_name[task_name].activation,
            )
            spec = plan.head_specs[task_name]
            artifacts.extend(
                _emit(
                    head_wrapper,
                    (spec.dummy_features.clone(),),
                    basename=output_dir / f"head_{task_name}",
                    input_names=resolve_export_io_names([spec.feature_key], prefix="input", generic=generic_io),
                    output_names=resolve_export_io_names([task_name], prefix="output", generic=generic_io),
                    kind="head",
                    config=config,
                    task_name=task_name,
                )
            )

    return artifacts
