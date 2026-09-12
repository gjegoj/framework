"""A built run, and what a run does: fit, evaluate, and end holding the weights it kept.

The order here is the whole of it, and it is deliberately readable in one screen — what a run does with
the objects the composition root handed it is a decision of the run section, not of any package.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from src.export import DeployableModel, Manifest, ship
from src.tracking import KeepsRecord
from src.training import load_checkpoint, restore_best_weights

if TYPE_CHECKING:
    import lightning as L

    from src.config import ExperimentConfig
    from src.export import Exporter
    from src.training import TrainingData, TrainingModule

MODEL = "model"
"""What a run's artifacts and the record describing them are named, under the run's own directory."""


@dataclass(frozen=True, slots=True)
class Experiment:
    """Everything a run needs, built and ready to start — including the declaration it was built from.

    The declaration travels with the run rather than beside it: it is what the run section asks for,
    what a tracker records as hyperparameters, and the same object the model was sized from. Two
    copies of it could disagree about which splits were prepared.
    """

    module: TrainingModule
    data: TrainingData
    trainer: L.Trainer
    declaration: ExperimentConfig
    exporters: Sequence[Exporter] = field(default_factory=tuple)


def run(experiment: Experiment) -> Manifest:
    """Fit, evaluate and ship, as the run section asks; answer with what was shipped.

    Only ``fit`` is ever handed a checkpoint, and only to continue an interrupted run; everything after
    it reads the module, so the weights evaluated are the ones the run actually stopped on.
    """
    config = experiment.declaration
    # Before anything can fail: what a tracker says about a run should not depend on the run finishing.
    # Lightning never does this itself — nothing here saves hyperparameters onto the module, because a
    # module holds built objects rather than the declaration they came from.
    if experiment.trainer.logger is not None:
        experiment.trainer.logger.log_hyperparams(config.model_dump(mode="json"))
    model = experiment.module.learner.model
    if config.run.checkpoint_path is not None:
        load_checkpoint(model, config.run.checkpoint_path)
    if config.run.train:
        experiment.trainer.fit(experiment.module, datamodule=experiment.data, ckpt_path=config.run.resume_path)
        restore_best_weights(experiment.trainer, model)
    if config.run.test:
        experiment.trainer.test(experiment.module, datamodule=experiment.data, verbose=False)
    return _ship(experiment)


def _ship(experiment: Experiment) -> Manifest:
    """Write the declared formats from the weights this run ended holding, and keep the record of it.

    Last of all, because everything before it can change which weights those are, and because writing
    moves the graph to the processor and back: a run with anything left to do would be doing it on a
    model that had just been moved twice for somebody else's benefit.

    The record goes to the tracker as well as beside the artifacts — the same one, where the run's
    declaration already went, so the two are read side by side. A backend with nowhere to keep one says
    so by not implementing the port, and the run carries on: the file beside the artifacts is the copy
    that exists whatever a run declared for a tracker, including nothing.
    """
    config = experiment.declaration
    learner = experiment.module.learner
    info = experiment.data.info
    graph = DeployableModel(learner.model, list(learner.tasks.values()), input_names=list(info.inputs))
    manifest = ship(graph, info, experiment.exporters, Path(config.run.directory) / MODEL)
    logger = experiment.trainer.logger
    if manifest.artifacts and isinstance(logger, KeepsRecord):
        logger.log_record(MODEL, manifest.as_record())
    return manifest
