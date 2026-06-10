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
    build_backbone,
    build_bindings,
    build_callbacks,
    build_data_module,
    build_lit_module,
    build_logger,
    build_optimizer_builder,
    build_tasks,
)
from src.config import load_config
from src.core.runtime import RuntimeContext
from src.models.assembly import build_composite_model
from src.training import LitDataModule
from src.utils.rich_utils import print_config

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    raw = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    print_config(raw)
    config = load_config(raw)

    L.seed_everything(config.seed, workers=True, verbose=False)

    # 1. Runtime context — populated incrementally as setup steps run
    runtime = RuntimeContext(epochs=config.epochs)

    # 2. Data: read → fit codecs (infers num_classes) → split → datasets
    bindings = build_bindings(config)
    plain_dm = build_data_module(config, bindings, runtime)
    plain_dm.setup()

    # 3. Tasks — built after setup so num_classes is a concrete int
    tasks = build_tasks(config, runtime)

    # 4. Model — heads sized from backbone.feature_dim, derived from tasks
    backbone = build_backbone(config.backbone)
    model = build_composite_model(backbone, {t.name: t.head_spec for t in tasks})

    # 5. Optimizer
    optimizer_builder = build_optimizer_builder(config.optimizer)

    # 6. Lightning wrappers (humble objects delegating to domain logic)
    lit_module = build_lit_module(config, model, tasks, optimizer_builder)
    lit_dm = LitDataModule(plain_dm)

    # 7. Fit
    logger = build_logger(config)
    callbacks = build_callbacks(config, runtime)
    trainer_kwargs = config.trainer.model_dump(mode="python")
    trainer_kwargs.pop("logger", None)
    trainer = L.Trainer(max_epochs=config.epochs, logger=logger, callbacks=callbacks, **trainer_kwargs)
    trainer.fit(lit_module, lit_dm)
    log.info("Training complete.")


if __name__ == "__main__":
    main()
