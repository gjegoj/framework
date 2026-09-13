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
from src.core import InputInfo, Stage
from src.data.build import build_data_module, build_preprocessor
from src.experiment import Experiment
from src.export.build import build_exporters
from src.losses.build import build_loss
from src.metrics.build import build_metrics
from src.models.build import build_model
from src.tasks.build import build_task_kinds, build_tasks, default_target_encoder, head_for
from src.tracking.build import build_tracker
from src.training import TrainingData, TrainingModule
from src.training.build import build_learner, build_optimizer_factory, build_profiler, build_scheduler_factory
from src.transforms import SampleTransform
from src.transforms.base import AppliesNormalization, is_the_same_scaling
from src.transforms.build import build_transforms

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.data import DataModule
    from src.losses import Loss
    from src.metrics import MetricCollection
    from src.tasks import Task

log = logging.getLogger(__name__)


def build(config: ExperimentConfig) -> Experiment:
    """Assemble a run from its declaration, in the one order the contracts allow.

    What needs no facts is built first, so a declaration that cannot hold is answered before the data is
    read; everything after that is ordered by what it needs, the data ahead of the model because a head
    is sized by what the data settled.
    """
    seed_everything(config.seed, workers=True)
    kinds = build_task_kinds(config.tasks)
    # Before the data, though it is used last: what a run ships is settled by its declaration alone, and
    # preparing the data is a source read, an encoder fit and a cache warm — the whole cost of a run that
    # is going to answer a misspelled format at the end of it.
    exporters = build_exporters(config.export)
    data = prepare_data(config, kinds)
    tasks = build_tasks(config.tasks, data.info)
    model = build_model(
        config.model,
        heads={name: head_for(config.tasks[name], task) for name, task in tasks.items()},
        outputs={name: task.output_shape() for name, task in tasks.items()},
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
        exporters=exporters,
    )


def prepare_data(config: ExperimentConfig, kinds: Mapping[str, type[Task]]) -> DataModule:
    """Read the sources, fit the encoders on the training split, and warm whatever cache there is.

    Eager on purpose, and here rather than inside a Lightning hook: every size the model is built from
    comes out of this, so it has to have happened before the model exists.
    """
    preprocessor = build_preprocessor(
        config.preprocessing,
        config.tasks,
        {name: default_target_encoder(kind) for name, kind in kinds.items()},
    )
    transforms = build_transforms(config.transforms, preprocessor.geometries)
    data = build_data_module(
        config.data,
        preprocessor=preprocessor,
        targets={name: declared.target_column for name, declared in config.tasks.items() if declared.target_column},
        transforms=transforms,
    )
    splits = needed_splits(config)
    data.setup(splits)
    if Stage.TRAIN in splits:
        data.fit_preprocessing(Stage.TRAIN)
    # After fitting rather than before: what an input declares is settled from the start, but the facts
    # are published together, and the encoder that learns its layout from the training split has not.
    _refuse_a_chain_that_does_not_apply_the_declared_scaling(data.info.inputs, transforms)
    data.warm(splits)
    return data


def _refuse_a_chain_that_does_not_apply_the_declared_scaling(
    inputs: Mapping[str, InputInfo], transforms: Mapping[str, SampleTransform]
) -> None:
    """An input's declared scaling has to be the one its stage's chain applies, or every reader of it lies.

    Two sections own the two halves: `preprocessing.inputs.<name>` says what the model was trained under,
    `transforms.<stage>` says what the pixels actually go through. Both are read elsewhere as fact — the
    manifest a deployment builds its input from, the sample page that undoes the scaling to show an image
    — and a run whose halves disagree makes both of them describe a model that never existed.

    A transform that cannot say what it applies is not asked: the capability is optional, and one a run
    wrote itself is its own business.
    """
    declared = {name: info.normalization for name, info in inputs.items() if info.normalization is not None}
    for stage, transform in transforms.items():
        if not isinstance(transform, AppliesNormalization):
            continue
        applied = transform.normalization
        for name, one in declared.items():
            if not is_the_same_scaling(one, applied):
                says = "no fixed scaling of its own" if applied is None else f"mean {applied.mean}, std {applied.std}"
                raise ValueError(
                    f"Input {name!r} is declared to be scaled by mean {one.mean}, std {one.std}, and the "
                    f"'{stage}' chain applies {says}. What reads the declaration — the record an export "
                    "ships, the page that undoes the scaling to show an image — would be describing a model "
                    "trained on something else. The shipped groups interpolate "
                    "`${preprocessing.inputs.<name>.mean}` into `albumentations.Normalize` so the two "
                    "cannot drift apart."
                )


def needed_splits(config: ExperimentConfig) -> tuple[Stage, ...]:
    """The splits this run will actually read: preparing one costs a read and a warm pass.

    Stages, because a stage and a split share a name — the convention ``Stage`` itself declares.

    A run that does not train reads neither the training split nor the validation one: it is evaluating
    or shipping weights it was handed, and reading a split to prepare it for nobody is the whole cost of
    the thing it is not doing. What such a run gives up is an encoder that learns its layout from the
    training data — it refuses by name instead, saying what to declare so it need not learn anything.
    """
    fitting = (Stage.TRAIN, Stage.VAL) if config.run.train else ()
    return (*fitting, *((Stage.TEST,) if config.run.test else ()))


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
