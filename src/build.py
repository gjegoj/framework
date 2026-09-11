"""The composition root: a validated declaration becomes a run that can be started.

Only wiring lives here — the one place that reads a whole ``ExperimentConfig`` and hands each package
the section it builds from. How a thing is *made* belongs to that package; what stays here is the order
things are made in, the facts that travel between them, and the handful of rules that no single section
can check because they are about two of them at once.

It reads top to bottom, and the order is itself a contract: the data is prepared before the model
exists, because that is what lets a head be sized by what the data settled rather than by what a file
repeats.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import lightning as L
from lightning import seed_everything

from src.callbacks.build import build_callbacks
from src.config import ExperimentConfig, TaskConfig
from src.core import Stage
from src.data.build import build_data_module, build_preprocessor, build_transforms
from src.experiment import Experiment
from src.losses.build import build_loss
from src.metrics.build import build_metrics
from src.models.build import build_model
from src.tasks.build import build_task_kinds, build_tasks, default_encoder, head_for
from src.tracking.build import build_tracker
from src.training import TrainingData, TrainingModule
from src.training.build import build_learner, build_optimizer_factory, build_profiler, build_scheduler_factory

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.data import DataModule
    from src.losses import Loss
    from src.metrics import MetricCollection
    from src.tasks import Task

log = logging.getLogger(__name__)


def build(config: ExperimentConfig) -> Experiment:
    """Assemble a run from its declaration, in the one order the contracts allow."""
    seed_everything(config.seed, workers=True)
    refuse_inputs_that_disagree(config)
    refuse_heads_a_whole_model_cannot_serve(config)
    kinds = build_task_kinds(config.tasks)
    data = prepare_data(config, kinds)
    tasks = build_tasks(config.tasks, data.info)
    model = build_model(
        config.model,
        heads={name: head_for(config.tasks[name], task) for name, task in tasks.items()},
        outputs={name: task.output_shape(task.info) for name, task in tasks.items()},
    )
    learner = build_learner(
        config.learner,
        model=model,
        tasks=tasks,
        losses={name: loss_for(config.tasks[name], task) for name, task in tasks.items()},
    )
    return Experiment(
        module=TrainingModule(
            learner,
            optimizer_factory=build_optimizer_factory(config.optimizer, config.lr),
            scheduler_factory=build_scheduler_factory(config.scheduler),
            metrics={name: metrics_for(config.tasks[name], task) for name, task in tasks.items()},
        ),
        data=TrainingData(data, batch_size=config.batch_size, **config.loader.model_dump()),
        trainer=build_trainer(config),
        declaration=config,
    )


def refuse_inputs_that_disagree(config: ExperimentConfig) -> None:
    """`data.inputs` binds a name to a column; `preprocessing.inputs` gives that name an encoder.

    Two declarations, one vocabulary. Left to itself the mismatch surfaces on the first batch, inside a
    loader worker, after the sources were read and the encoders fitted — so the two key sets are
    compared here, in the one place that sees both.
    """
    bound = set(config.data.params.get("inputs", {}))
    encoded = set(config.preprocessing.inputs or {})
    if bound and bound != encoded:
        raise ValueError(
            f"The inputs a run binds to columns and the inputs it encodes are different names: "
            f"data.inputs has {', '.join(sorted(bound)) or 'none'}, preprocessing.inputs has "
            f"{', '.join(sorted(encoded)) or 'none'}. They name the same values."
        )


def refuse_heads_a_whole_model_cannot_serve(config: ExperimentConfig) -> None:
    """A model reached by import path arrives whole — head, decoding and all — so it composes none.

    The two sections are declared apart and only here are both in view. Silently ignoring a declared
    head would leave a run reporting numbers for a recipe nobody ran.
    """
    if config.model.import_path is None:
        return
    declared = sorted(name for name, task in config.tasks.items() if task.head is not None)
    if declared:
        raise ValueError(
            f"{config.model.spelled!r} is a whole model and brings its own heads, so the head declared for "
            f"{', '.join(declared)} would never be built. Drop it, or declare a backbone to compose onto."
        )


def prepare_data(config: ExperimentConfig, kinds: Mapping[str, type[Task]]) -> DataModule:
    """Read the sources, fit the encoders on the training split, and warm whatever cache there is.

    Eager on purpose, and here rather than inside a Lightning hook: every size the model is built from
    comes out of this, so it has to have happened before the model exists.
    """
    preprocessor = build_preprocessor(
        config.preprocessing,
        config.tasks,
        {name: default_encoder(kind) for name, kind in kinds.items()},
    )
    data = build_data_module(
        config.data,
        preprocessor=preprocessor,
        targets={name: task.target for name, task in config.tasks.items() if task.target},
        transforms=build_transforms(config.transforms, preprocessor.geometries),
    )
    splits = needed_splits(config)
    data.setup(splits)
    data.fit_preprocessing(Stage.TRAIN)
    data.warm(splits)
    return data


def needed_splits(config: ExperimentConfig) -> tuple[Stage, ...]:
    """The splits this run will actually read: preparing one costs a read and a warm pass.

    Stages, because a stage and a split share a name — the convention ``Stage`` itself declares.
    """
    return (Stage.TRAIN, Stage.VAL, *((Stage.TEST,) if config.run.test else ()))


def loss_for(declared: TaskConfig, task: Task) -> Loss:
    """What a task is learned by: what the run declared, or what its kind implies from the facts.

    Here rather than in either package: ``losses`` knows nothing of tasks and ``tasks`` knows nothing of
    losses, so the root is the only place holding both a declaration and the facts to size it from.
    """
    return build_loss(declared.loss if declared.loss is not None else task.default_loss, task.facts())


def metrics_for(declared: TaskConfig, task: Task) -> MetricCollection:
    """What a task is judged by; a declared set replaces the kind's own rather than adding to it."""
    if declared.metrics is None:
        log.info("Task %r is judged by its kind's own metrics: %s.", task.name, ", ".join(task.default_metrics))
    return build_metrics(declared.metrics if declared.metrics is not None else task.default_metrics, task.facts())


def build_trainer(config: ExperimentConfig) -> L.Trainer:
    """The loop itself: how long it runs, what it records to, and what runs alongside it.

    ``logger=False`` rather than None where a run declares no tracker: left to itself Lightning starts
    a logger of its own, which is not what `tracker: none` says. Where a run's files land is written in
    config as ``${run.directory}``, because Lightning would otherwise resolve it from the tracker.
    """
    tracker = build_tracker(config.tracker)
    return L.Trainer(
        max_epochs=config.epochs,
        default_root_dir=config.run.directory,
        logger=tracker if tracker is not None else False,
        callbacks=build_callbacks(config.callbacks),
        profiler=build_profiler(config.trainer.profiler),
        **config.trainer.model_dump(exclude={"profiler"}),
    )
