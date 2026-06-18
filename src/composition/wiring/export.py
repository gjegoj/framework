"""Export wiring: resolve output dir and invoke the export pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import lightning as L

from src.composition.wiring.checkpointing import load_init_weights, resolve_ckpt_file, resolve_test_ckpt_path
from src.config.schema import ExperimentConfig
from src.core.entities import Task
from src.export.pipeline import export_model
from src.export.spec import guard_exportable_topologies
from src.training.modules import LitModule

log = logging.getLogger(__name__)


def ensure_module_weights_for_export(
    trainer: L.Trainer,
    lit_module: LitModule,
    config: ExperimentConfig,
    *,
    trained: bool,
    tested: bool,
) -> None:
    """Load checkpoint weights when export runs without a preceding ``test`` call.

    Parameters:
        trainer (L.Trainer): Trainer (for checkpoint callback paths).
        lit_module (LitModule): Module to load into.
        config (ExperimentConfig): Experiment config.
        trained (bool): Whether ``fit`` ran in this session.
        tested (bool): Whether ``test`` ran in this session.
    """
    if tested:
        return
    ckpt_path = resolve_test_ckpt_path(trainer, config, trained=trained)
    if ckpt_path is None:
        return
    load_init_weights(lit_module, resolve_ckpt_file(trainer, ckpt_path))


def validate_export_preconditions(config: ExperimentConfig, tasks: list[Task]) -> None:
    """Fail-fast export checks that need only config + tasks (run before training).

    Per-format option validation already happens in ``load_config`` (the
    ``ExportConfig`` discriminated union). This adds the model-independent
    topology guard so an unexportable task is caught before a training run is
    spent. Heavy, weight-dependent checks (the dummy backbone forward) stay in
    ``build_export_plan`` at export time.

    Parameters:
        config (ExperimentConfig): Validated experiment config.
        tasks (list[Task]): Active tasks in declaration order.
    """
    if not config.run_export:
        return
    guard_exportable_topologies(tasks)


def run_export(
    trainer: L.Trainer,
    lit_module: LitModule,
    tasks: list[Task],
    config: ExperimentConfig,
    *,
    trained: bool,
    tested: bool,
) -> None:
    """Export deployment artifacts when ``run_export`` is enabled.

    Parameters:
        trainer (L.Trainer): Trainer instance.
        lit_module (LitModule): Trained/evaluated module.
        tasks (list[Task]): Active tasks.
        config (ExperimentConfig): Validated config.
        trained (bool): Whether training ran.
        tested (bool): Whether testing ran.
    """
    if not config.run_export or not config.export.targets:
        return

    ensure_module_weights_for_export(trainer, lit_module, config, trained=trained, tested=tested)

    if config.export.output_dir is not None:
        output_dir = Path(config.export.output_dir)
    elif config.save_dir is not None:
        output_dir = Path(config.save_dir) / "export"
    else:
        output_dir = Path("export")

    lit_module.eval()
    artifacts = export_model(lit_module.model, tasks, config, output_dir)
    log.info("Export complete: %d artifact(s) in %s.", len(artifacts), output_dir)
