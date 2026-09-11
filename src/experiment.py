"""A built run, and what a run does: fit, evaluate, and end holding the weights it kept.

The order here is the whole of it, and it is deliberately readable in one screen — what a run does with
the objects the composition root handed it is a decision of the run section, not of any package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.training import load_checkpoint, restore_best_weights

if TYPE_CHECKING:
    import lightning as L

    from src.config import ExperimentConfig
    from src.training import TrainingData, TrainingModule


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


def run(experiment: Experiment) -> None:
    """Fit and evaluate, as the run section asks.

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
