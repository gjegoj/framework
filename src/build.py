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
from src.data.build import build_data_module, build_preprocessor
from src.experiment import Experiment
from src.export import WRITTEN_FROM, example_inputs
from src.export.build import build_exporters
from src.losses.build import build_loss
from src.metrics.build import build_metrics
from src.models.build import build_adapter, build_model
from src.tasks.build import build_task_kinds, build_tasks, default_target_encoder, head_for
from src.tracking import MetricKey
from src.tracking.build import build_tracker
from src.training import LOSS, TrainingData, TrainingModule
from src.training.build import (
    build_learner,
    build_optimizer_factory,
    build_profiler,
    build_scheduler_factory,
    build_teacher,
)
from src.transforms.build import build_transforms

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from src.core import DatasetInfo
    from src.data import DataModule
    from src.export import Exporter
    from src.losses import Loss
    from src.metrics import MetricCollection
    from src.models import Model
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
    _refuse_a_run_that_could_never_write_what_it_declares(exporters, data.info)
    tasks = build_tasks(config.tasks, data.info)
    # Named here rather than inline, because a second network is built from the very same two: a teacher
    # answers the questions this run asks, so its heads are sized by what the tasks settled, not by a file.
    heads = {name: head_for(config.tasks[name], task) for name, task in tasks.items()}
    outputs = {name: task.output_shape() for name, task in tasks.items()}
    model = build_model(config.model, heads=heads, outputs=outputs)
    # Between building the network and training it, because that is the whole of what an adapter is: the
    # weights come from wherever the model section says, and what this run learns is added beside them.
    adapter = build_adapter(config.adapter, model)
    losses = {name: loss_for(config.tasks[name], task) for name, task in tasks.items()}
    measured = {name: metrics_for(config.tasks[name], task) for name, task in tasks.items()}
    _refuse_a_head_and_an_objective_that_disagree(model, losses)
    _refuse_watching_an_objective_this_run_never_scores(config, tasks, losses, measured)
    learner = build_learner(
        config.learner,
        model=model,
        tasks=tasks,
        losses=losses,
        teacher=build_teacher(config.teacher, heads=heads, outputs=outputs),
    )
    return Experiment(
        module=TrainingModule(
            learner,
            optimizer_factory=build_optimizer_factory(config.optimizer, config.lr),
            scheduler_factory=build_scheduler_factory(config.scheduler),
            metrics=measured,
        ),
        data=TrainingData(data, batch_size=config.batch_size, **config.loader.model_dump()),
        trainer=build_trainer(config),
        declaration=config,
        exporters=exporters,
        adapter=adapter,
    )


def _refuse_a_run_that_could_never_write_what_it_declares(exporters: Sequence[Exporter], info: DatasetInfo) -> None:
    """An export is written from an example of the declared inputs; a run that cannot have one is told here.

    The check *is* the operation — the very call shipping makes, on the very facts it makes it from —
    so there is no second statement of what an example needs, free to fall behind the first. What it
    costs is one batch of noise; what it saves is hearing after the last epoch that nothing can be
    written from it, which is where this was answered before.
    """
    if exporters:
        example_inputs(info, list(info.inputs), WRITTEN_FROM)


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
    data.warm(splits)
    return data


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


def _refuse_a_head_and_an_objective_that_disagree(model: Model, losses: Mapping[str, Loss]) -> None:
    """The network and the objective over it are built apart and have to agree about one tensor.

    Here because only the root holds both. Why neither the shape nor the values tell a projection and
    an angle apart is ``Representation``, which is the word the two declare in.

    Compared by value rather than by identity: ``Representation`` is a ``StrEnum`` so that a head a run
    wrote itself may spell ``produces = "cosines"`` and be taken at its word.
    """
    for name, loss in losses.items():
        answered = model.produces(name)
        if loss.reads != answered:
            raise ValueError(
                f"Task {name!r}: the network serving it answers with {answered}, and objective "
                f"{loss.log_name!r} reads {loss.reads}. Nothing in a tensor says which of the two it "
                f"holds, so this pair would train and report a number that looks like work. Declare "
                f"`tasks.{name}.loss` that reads {answered}, or a head that answers with {loss.reads} — "
                f"`tasks.{name}.head` where the run composes one, `produces` on a network arriving whole."
            )


def _refuse_watching_an_objective_this_run_never_scores(
    config: ExperimentConfig,
    tasks: Mapping[str, Task],
    losses: Mapping[str, Loss],
    measured: Mapping[str, MetricCollection],
) -> None:
    """A run is kept by the number it is watched by, and some of them it will never write.

    Here because only the root holds both sides: which objectives stop outside training — a task judged
    on a vocabulary the training split settled has one, and ``build_learner`` reads the same fact — and
    what the declaration asked to be watched. Both watchers are held to it, the saver that keeps an
    epoch and the schedule that reacts to one, because it is one question asked twice.

    Lightning refuses this itself, and not badly: measured, ``MisconfigurationException`` at the end of
    the first validation, listing the keys that do exist. What this buys is when and what — while the run
    is still being assembled rather than an epoch into it, and naming the readings to watch instead,
    which a library that knows nothing of retrieval cannot.

    Not earlier than that, and the reason is the order above: which objectives stop is read off what the
    encoders settled, so this cannot come before the data is prepared. Measured on the shipped example
    over 256 rows, the refusal lands at 5.9 s against a training epoch spent before Lightning's — and
    what stands between the two on a real dataset is every minute of that epoch.

    Only keys that certainly will not exist. The full set a run logs is not knowable here: a per-class
    metric's leaves appear when it computes, and a composite objective's terms come out of its own
    breakdown. This answers the narrower question — a stage in which nothing is scored writes no
    objective at all — and says nothing about a misspelled metric, which Lightning still catches late.
    """
    if not all(task.info.open_set for task in tasks.values()):
        return
    unwritten = {str(MetricKey(stage, LOSS)) for stage in (Stage.VAL, Stage.TEST)} | {
        str(MetricKey(stage, loss.log_name, task=name))
        for stage in (Stage.VAL, Stage.TEST)
        for name, loss in losses.items()
    }
    watchers = [("callbacks", one.params.get("monitor")) for one in config.callbacks] + [
        ("scheduler", config.scheduler.monitor if config.scheduler else None)
    ]
    for section, watched in watchers:
        if watched in unwritten:
            raise ValueError(
                f"`{section}` watches {watched!r}, and this run never writes it: {', '.join(sorted(tasks))} "
                f"is judged on a vocabulary the training split settled, so its objective is scored while "
                f"it is being learned and nowhere else. Keep the run by what evaluation does measure"
                f"{_watchable_in_evaluation(measured)}, or pin the vocabulary with "
                f"`tasks.<name>.target_encoder: label` and `tasks.<name>.classes`, and the objective is "
                f"then scored in every stage."
            )


def _watchable_in_evaluation(measured: Mapping[str, MetricCollection]) -> str:
    """What the refusal above offers instead: each reading a run could be kept by, and which way.

    The direction as well as the key, because a key alone is half an instruction and the shipped saver
    watches with ``mode: min`` — a reader who swaps only the key keeps the epoch that scored *worst*,
    which is the defect being refused wearing different clothes. Read off the metrics themselves, which
    is where a direction is declared and the same answer ``TrainingModule.metric_directions`` reports.

    A reading that declares no direction is not offered at all. It is measured in evaluation and still
    not an answer to this question: kept by a verification threshold, a run would choose the epoch whose
    separation drifted furthest from the rest.

    Answers with the clause that goes into the sentence, empty where a run measures nothing it could be
    kept by — the offer is then simply not made, rather than made of nothing.
    """
    by_direction: dict[str, list[str]] = {}
    for name, collection in measured.items():
        for label, metric in collection.items():
            if metric.higher_is_better is not None:
                mode = "max" if metric.higher_is_better else "min"
                by_direction.setdefault(mode, []).append(f"`{MetricKey(Stage.VAL, label, task=name)}`")
    offered = "; ".join(f"{', '.join(keys)} with `mode: {mode}`" for mode, keys in sorted(by_direction.items()))
    return f" — {offered}" if offered else ""


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
