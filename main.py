"""Training entry point.

Usage:
    python main.py                          # uses defaults (experiment: classification_smoke)
    python main.py +experiment=my_exp       # load a specific experiment override
    python main.py epochs=5 batch_size=32   # ad-hoc CLI overrides
    python main.py run_test=false           # train only, skip test
    python main.py run_train=false run_test=true \\
        ckpt_path=runs/.../checkpoints/epoch=11.ckpt   # eval-only on a checkpoint
    python main.py init_ckpt_path=runs/.../epoch=3.ckpt  # fine-tune from pretrained weights

Hydra writes run outputs to outputs/<date>/<time>/. Override with
hydra.run.dir=<path> or add hydra/output: null to suppress.
"""

from __future__ import annotations

import logging

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

import src.models  # noqa: F401 — populate the backbone / head registries
import src.tasks  # noqa: F401 — populate the topology / objective / preset (and criteria) registries
from src.composition.wiring import (
    build_backbone,
    build_bindings,
    build_callbacks,
    build_data_module,
    build_lit_data_module,
    build_lit_module,
    build_logger,
    build_optimizer_builder,
    build_scheduler_builder,
    build_task_lr_overrides,
    build_tasks,
    build_trainer,
    run_experiment,
    validate_export_preconditions,
)
from src.config import load_config
from src.core.runtime import RuntimeContext
from src.models.assembly import build_composite_model
from src.utils.console import print_config, silence_known_warnings

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(hydra_config: DictConfig) -> None:
    silence_known_warnings()
    raw = OmegaConf.to_container(hydra_config, resolve=True, throw_on_missing=True)
    print_config(raw)
    config = load_config(raw)

    L.seed_everything(config.seed, workers=True, verbose=False)

    # 1. Runtime context — populated incrementally as setup steps run
    runtime = RuntimeContext()

    # 2. Data: read → fit encoders (infers num_classes) → split → datasets
    bindings = build_bindings(config)
    plain_data_module = build_data_module(config, bindings, runtime)
    plain_data_module.setup()

    # 3. Tasks — built after setup so num_classes is a concrete int
    tasks = build_tasks(config, runtime)
    validate_export_preconditions(config, tasks)  # fail before training if export is impossible

    # 4. Model — heads sized from backbone.feature_dim, derived from tasks
    backbone = build_backbone(config.backbone)
    model = build_composite_model(backbone, {task.name: task.head_spec for task in tasks})

    # 5. Optimizer — per-head LR overrides (from task configs) bound into the builder
    optimizer_builder = build_optimizer_builder(config.optimizer, build_task_lr_overrides(config))
    scheduler_builder = build_scheduler_builder(config.scheduler)

    # 6. Lightning wrappers (humble objects delegating to domain logic)
    lit_module = build_lit_module(config, model, tasks, optimizer_builder, scheduler_builder)
    lit_data_module = build_lit_data_module(plain_data_module)

    # 7. Train and/or test
    logger = build_logger(config)
    callbacks = build_callbacks(config, runtime)
    trainer = build_trainer(config, logger, callbacks)
    run_experiment(trainer, lit_module, lit_data_module, config, tasks)


if __name__ == "__main__":
    main()
