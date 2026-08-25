"""The composition root proper: a validated config becomes a running experiment."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lightning import seed_everything

from src.assembly.checkpoints import load_weights
from src.assembly.data import build_data_module
from src.assembly.export import build_exporters, export_model
from src.assembly.models import build_model
from src.assembly.tasks import refuse_what_the_composite_family_cannot_serve
from src.assembly.training import (
    build_optimizer_factory,
    build_scheduler_factory,
    build_trainer,
    build_training_data,
)
from src.assembly.vendor import refuse_what_a_vendor_cannot_serve
from src.core.entities import DataProfile
from src.models import merge_adapters, without_teachers
from src.training import TrainingModule

if TYPE_CHECKING:
    import lightning as L
    from torch import nn

    from src.config import ExperimentConfig
    from src.core.entities import Task
    from src.export import Exporter
    from src.training import TrainingData

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Experiment:
    """Everything a run needs, assembled and ready."""

    module: TrainingModule
    data: TrainingData
    trainer: L.Trainer
    tasks: list[Task]
    exporters: list[Exporter]


def assemble(config: ExperimentConfig) -> Experiment:
    """Build an experiment from config, in the one order the contract allows.

    What a vendor family cannot serve is refused first, before a dataset is read or a
    network is built — and so is a task no composed backbone can serve: an hour of
    training is a long way to carry a section that was never going to apply.

    ``DataModule.setup`` runs before the model is built: that ordering is what
    lets output sizes come from data instead of from config. It is universal
    across model families — a vendor data module writes its own facts into the
    same profile — which is why it stays visible here rather than hiding inside
    each family. No concrete family is named in this function.
    """
    refuse_what_a_vendor_cannot_serve(config)
    refuse_what_the_composite_family_cannot_serve(config)
    seed_everything(config.seed, workers=True)
    data_module = build_data_module(config)
    profile = DataProfile()
    data_module.setup(profile)
    model, tasks = build_model(config, profile)
    module = TrainingModule(
        model=model,
        tasks=tasks,
        optimizer_factory=build_optimizer_factory(config),
        scheduler_factory=build_scheduler_factory(config),
    )
    return Experiment(
        module=module,
        data=build_training_data(config, data_module),
        trainer=build_trainer(config, tasks, profile, architecture=model.architecture),
        tasks=tasks,
        exporters=build_exporters(config),
    )


def run(experiment: Experiment, config: ExperimentConfig) -> None:
    """Fit, judge and ship, as the run section asks.

    Only ``fit`` is ever handed a checkpoint, and only to continue an interrupted
    run. Everything after it reads the module, so the weights that are judged and
    the weights that are shipped are the ones the run actually stopped on —
    measured, handing the same file to ``test`` reloads it over whatever training
    produced.

    ``shipped`` is what the run is about: with distillation the module also holds
    frozen teachers, and they are neither loaded into, nor folded, nor exported.
    """
    # Before anything can fail: what a tracker shows about a run should not depend on
    # the run finishing. Lightning never calls this itself — nothing here saves
    # hyperparameters onto the module, because a module holds built objects, not config.
    if experiment.trainer.logger is not None:
        experiment.trainer.logger.log_hyperparams(config.model_dump(mode="json"))
    shipped = without_teachers(experiment.module.model)
    if config.run.checkpoint_path is not None:
        load_weights(shipped, config.run.checkpoint_path)
    if config.run.train:
        experiment.trainer.fit(experiment.module, datamodule=experiment.data, ckpt_path=config.run.resume_path)
        restore_best_weights(experiment.trainer, shipped)
    # After the run's weights are settled and before anything reads them: a checkpoint is
    # keyed for the adapted model, and the artifact must carry none of the adapters' overhead.
    merge_adapters(shipped)
    if config.run.test:
        experiment.trainer.test(experiment.module, datamodule=experiment.data, verbose=False)
    if experiment.exporters:
        export_model(shipped, config, experiment.exporters)


def restore_best_weights(trainer: L.Trainer, model: nn.Module) -> None:
    """Put the checkpoint the run kept back into the model.

    Lightning does not: measured, ``_CheckpointConnector._parse_ckpt_path``
    returns ``None`` whenever the module is passed explicitly, so a run with a
    monitor would report the last epoch's numbers while keeping a different
    epoch on disk — and then ship that last epoch. A run that kept nothing has
    an empty ``best_model_path`` and this is a no-op.
    """
    checkpoint = trainer.checkpoint_callback
    path = str(getattr(checkpoint, "best_model_path", "") or "")
    if not path:
        return
    load_weights(model, path)
