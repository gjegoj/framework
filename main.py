"""Training entry point.

Usage:
    python main.py                          # uses defaults (experiment: classification_smoke)
    python main.py +experiment=my_exp       # load a specific experiment override
    python main.py epochs=5 batch_size=32   # ad-hoc CLI overrides

Hydra writes run outputs to outputs/<date>/<time>/. Override with
hydra.run.dir=<path> or add hydra/output: null to suppress.
"""

from __future__ import annotations

import logging

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

import src.models  # noqa: F401 — registers TimmBackbone and LinearHead
import src.tasks  # noqa: F401 — registers topology/objective strategies and presets
from src.composition.wiring import (
    build_bindings,
    build_optimizer_builder,
    build_tasks,
    build_transforms,
)
from src.config import load_config
from src.core.runtime import RuntimeContext
from src.data import CsvDataSource, DataModule
from src.models.assembly import build_composite_model
from src.models.registry import backbones
from src.training import LitDataModule, LitModule

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    raw = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    config = load_config(raw)  # type: ignore[arg-type]

    L.seed_everything(config.seed, workers=True)
    log.info(
        "Project: %s | seed: %d | epochs: %d",
        config.project,
        config.seed,
        config.epochs,
    )

    # 1. Runtime context — populated incrementally as setup steps run
    runtime = RuntimeContext(epochs=config.epochs)

    # 2. Data: read → fit codecs (infers num_classes) → split → datasets
    sources = config.data.source if isinstance(config.data.source, list) else [config.data.source]
    bindings = build_bindings(config)
    plain_dm = DataModule(
        source=CsvDataSource(sources),
        bindings=bindings,
        image_column=config.data.image_column,
        transforms=build_transforms(config),
        split=config.data.split,
        runtime=runtime,
        batch_size=config.batch_size,
        seed=config.seed,
        root_path=config.data.root_path,
    )
    plain_dm.setup()
    log.info("Dataset sizes: %s", dict(runtime.dataset_sizes))
    log.info("Inferred num_classes: %s", dict(runtime.num_classes))

    # 3. Tasks — built after setup so num_classes is a concrete int
    tasks = build_tasks(config, runtime)
    log.info("Tasks: %s", [t.name for t in tasks])

    # 4. Model — heads sized from backbone.feature_dim, derived from tasks
    backbone = backbones.create(
        config.backbone.kind,
        name=config.backbone.name,
        pretrained=config.backbone.pretrained,
    )
    model = build_composite_model(backbone, {t.name: t.head_spec for t in tasks})

    # 5. Optimizer (class resolved from optimizer.name) with optional per-head LR overrides
    task_lr_overrides = {
        name: task_cfg.optimizer.lr for name, task_cfg in config.tasks.items() if task_cfg.optimizer is not None
    }
    optimizer_builder = build_optimizer_builder(config.optimizer)

    # 6. Lightning wrappers (humble objects delegating to domain logic)
    lit_module = LitModule(
        model=model,
        tasks=tasks,
        optimizer_builder=optimizer_builder,
        task_lr_overrides=task_lr_overrides or None,
    )
    lit_dm = LitDataModule(plain_dm)

    # 7. Fit
    trainer_kwargs = config.trainer.model_dump(mode="python")
    trainer = L.Trainer(max_epochs=config.epochs, **trainer_kwargs)
    trainer.fit(lit_module, lit_dm)
    log.info("Training complete.")


if __name__ == "__main__":
    main()
