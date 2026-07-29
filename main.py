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

import src.models
import src.tasks  # noqa: F401 — populate the topology / objective / preset (and criteria) registries
from src.composition.wiring import (
    build_callbacks,
    build_logger,
    build_trainer,
    resolve_experiment_assembler,
    run_experiment,
)
from src.config import load_config
from src.core.runtime import RuntimeContext
from src.utils.console import print_config, silence_known_warnings

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(hydra_config: DictConfig) -> None:
    silence_known_warnings()
    raw = OmegaConf.to_container(hydra_config, resolve=True, throw_on_missing=True)
    print_config(raw)
    config = load_config(raw)

    L.seed_everything(config.seed, workers=True, verbose=False)
    runtime = RuntimeContext()

    # One assembler per run family (standard chain, detection, ...) — resolved from
    # config, validated up front, then built. Adding a family never touches this file.
    assembler = resolve_experiment_assembler(config)
    assembler.validate(config)
    lit_module, lit_data_module, tasks = assembler.build(config, runtime)

    # Shared tail: logging, callbacks, trainer, fit/test/export.
    logger = build_logger(config)
    callbacks = build_callbacks(config, runtime)
    trainer = build_trainer(config, logger, callbacks)
    run_experiment(trainer, lit_module, lit_data_module, config, tasks)


if __name__ == "__main__":
    main()
