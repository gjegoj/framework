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
import numpy as np
from lightning import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from src.callbacks import EmaWeights, Freeze
from src.callbacks.build import build_callbacks
from src.config import ExperimentConfig, HeadConfig, TaskConfig
from src.core import Axis, Geometry, Sample, Stage, TensorShape, naming, require_tensor
from src.data.build import build_data_module, build_preprocessor
from src.experiment import Experiment
from src.export import WRITTEN_FROM, example_inputs
from src.export.build import build_exporters
from src.losses.build import build_loss, refuse_an_objective_the_head_does_not_answer
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
    refuse_a_learner_and_its_child_positions_that_disagree,
)
from src.transforms.build import build_transforms

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lightning.pytorch.callbacks import Callback
    from lightning.pytorch.loggers import Logger

    from src.core import DatasetInfo, Normalization
    from src.data import DataModule
    from src.export import Exporter
    from src.losses import Loss
    from src.metrics import MetricCollection
    from src.models import Model
    from src.tasks import Task
    from src.transforms import SampleTransform

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
    # Built here, before the data, for what they answer rather than for what they do: three sections
    # mean nothing without another, and which class a declaration names is the only honest way to ask.
    # Nothing is started by building them — a tracker opens its run on the first thing reported to it.
    tracker = build_tracker(config.tracker)
    callbacks = build_callbacks(config.callbacks)
    _refuse_a_rate_watched_with_nothing_recording(tracker, callbacks)
    _refuse_an_average_no_checkpoint_would_keep(callbacks)
    _refuse_freezing_what_this_run_adapts(config, callbacks)
    refuse_a_learner_and_its_child_positions_that_disagree(config.learner)
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
    _refuse_a_head_of_several_streams_under_an_objective_that_reads_one(heads, losses)
    _refuse_watching_an_objective_this_run_never_scores(config, tasks, losses, measured)
    learner = build_learner(
        config.learner,
        model=model,
        tasks=tasks,
        losses=losses,
        teacher=build_teacher(config.learner.teacher, heads=heads, outputs=outputs),
    )
    return Experiment(
        module=TrainingModule(
            learner,
            optimizer_factory=build_optimizer_factory(config.optimizer, config.lr),
            scheduler_factory=build_scheduler_factory(config.scheduler),
            metrics=measured,
        ),
        data=TrainingData(data, batch_size=config.batch_size, **config.loader.model_dump()),
        trainer=build_trainer(config, tracker, callbacks),
        declaration=config,
        exporters=exporters,
        adapter=adapter,
    )


def _refuse_a_rate_watched_with_nothing_recording(tracker: Logger | None, callbacks: Sequence[Callback]) -> None:
    """A callback that only reports needs somewhere to report to, and the two are separate sections.

    Left alone, Lightning refuses this itself — but at ``on_train_start``, after the sources have been
    read, the encoders fitted and the cache warmed, and in words naming ``LearningRateMonitor``, the
    ``Trainer`` and its ``logger``: three things that appear nowhere in what the run declared.
    """
    if tracker is None and any(isinstance(one, LearningRateMonitor) for one in callbacks):
        raise ValueError(
            "callbacks holds a LearningRateMonitor and tracker is none, so there is nowhere to write a "
            "learning rate. Either declare a tracker — `tracker=csv` keeps the numbers in the run's own "
            "directory — or run without the callbacks that report: `callbacks=none`. A list cannot be "
            "edited from the command line, because Hydra will not force-add to a group."
        )


def _refuse_an_average_no_checkpoint_would_keep(callbacks: Sequence[Callback]) -> None:
    """An average of the weights and the file meant to hold it are two sections, and one can miss the other.

    Measured on lightning 2.6.5: a callback's ``on_save_checkpoint`` runs only for full checkpoints, so
    on the weights-only path the average is never substituted into the file. Nothing refuses the pair,
    because each section is right on its own — and the run keeps a file of live weights chosen by a
    metric read off the averaged ones, with every number it printed about them ordinary.

    Here rather than where the averaging begins, which is the earliest moment *that* callback can see
    the trainer's checkpoints — and by then the sources are read, the encoders fitted and the cache
    warmed. This list is the whole of what can refuse: what Lightning adds to it itself is one default
    ``ModelCheckpoint``, only where a run declared none, and its ``save_weights_only`` is false.
    """
    if any(isinstance(one, EmaWeights) for one in callbacks) and any(
        isinstance(one, ModelCheckpoint) and one.save_weights_only for one in callbacks
    ):
        raise ValueError(
            "An average of the weights cannot be kept by a checkpoint declaring save_weights_only: "
            "the file would hold the live weights while the metric it was chosen by came from the "
            "averaged ones. Declare save_weights_only: false — a full checkpoint is what "
            "`run.resume_path` continues from anyway."
        )


def _refuse_freezing_what_this_run_adapts(config: ExperimentConfig, callbacks: Sequence[Callback]) -> None:
    """Held still, a delta never moves, and the run trains its heads alone while the declaration says otherwise.

    Measured on a resnet adapted at its convolutions: with ``freeze`` over the same module, none of the
    thirty-four tensors of the delta are left learning, and nothing anywhere says so — the run trains,
    logs, keeps an epoch and ships it. The pair is redundant at best, because attaching a delta already
    holds the weights beneath it still; what it costs at worst is the whole run.

    The freeze is the built callback and the adapted module is the declaration's own: ``module`` is a
    parameter of ``Adapter`` itself, so the declaration is where it is typed and owned, while what a
    callback *is* cannot be read off a declaration at all.
    """
    adapted = str(config.adapter.params.get("module", "")) if config.adapter is not None else ""
    if not adapted:
        return
    for one in callbacks:
        if not isinstance(one, Freeze):
            continue
        if held := [path for path in one.modules if _reaches(path, adapted)]:
            raise ValueError(
                f"`adapter` adds parameters under {adapted!r} and `callbacks` freezes "
                f"{', '.join(repr(path) for path in held)}: the delta would be held still along with the "
                f"weights it was added to, and nothing under {adapted!r} would learn at all. Attaching a "
                f"delta already holds those weights still — drop the freeze, or freeze parts the adapter "
                f"does not reach."
            )


def _reaches(frozen: str, adapted: str) -> bool:
    """Whether holding one dot-path still holds the other: the same module, or either one inside the other."""
    return frozen == adapted or frozen.startswith(f"{adapted}.") or adapted.startswith(f"{frozen}.")


def _refuse_a_run_that_could_never_write_what_it_declares(exporters: Sequence[Exporter], info: DatasetInfo) -> None:
    """An export is written from an example of the declared inputs; a run that cannot have one is told here.

    The check *is* the operation — the very call shipping makes, on the very facts it makes it from —
    so there is no second statement of what an example needs, free to fall behind the first. What it
    costs is one batch of noise; what it saves is hearing after the last epoch that nothing can be
    written from it, which is where this was answered before.
    """
    if exporters:
        example_inputs(info, list(info.inputs), WRITTEN_FROM)


FULL = 255.0
"""What an 8-bit pixel reads at its brightest, which is the range a chain is asked to scale from."""

SHADES = (32.0, 224.0)
"""The darkest and brightest level the probe uses; inside the range so that no clipping hides an error."""

TOLERANCE = 1e-3
"""How far a resize's own interpolation may move a flat image before the scaling is called wrong."""


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
    splits = needed_splits(config)
    _refuse_a_split_this_run_reads_with_nothing_declared_to_prepare_it(preprocessor.geometries, transforms, splits)
    data = build_data_module(
        config.data,
        preprocessor=preprocessor,
        targets={name: declared.target_column for name, declared in config.tasks.items() if declared.target_column},
        transforms=transforms,
    )
    data.setup(splits)
    if Stage.TRAIN in splits:
        data.fit_preprocessing(Stage.TRAIN)
    # After the fit and before the warm: an encoder that learns its layout has nothing to publish until
    # the first of those, and the second is the expensive one — a run whose pixels do not become what it
    # declared should hear so before it reads a dataset, not after.
    _refuse_a_chain_that_does_not_scale_as_the_declaration_promises(data.info, transforms)
    data.warm(splits)
    return data


def _refuse_a_split_this_run_reads_with_nothing_declared_to_prepare_it(
    geometries: Mapping[str, Mapping[str, Geometry]],
    transforms: Mapping[str, SampleTransform],
    splits: Sequence[Stage],
) -> None:
    """A stage whose rows this run will read, and nothing declared to turn their pixels into tensors.

    A missing key in ``transforms`` is simply no chain, which is the right reading for a run over values
    that do not move: a text run declares ``transforms: {}`` and means it. It is the wrong reading for
    anything with a picture in it, and nothing said so — such a row reaches the encoders as the array its
    file decoded to, and is refused one sample at a time inside a loader worker, at the first batch of
    that stage. For ``test`` that is after the whole fit, and the refusal names the input rather than the
    stage, which is the one thing the reader would have to go and edit.
    """
    moved = sorted(
        {name for cells in geometries.values() for name, geometry in cells.items() if geometry is not Geometry.NONE}
    )
    missing = [stage for stage in splits if stage not in transforms]
    if not moved or not missing:
        return
    named = ", ".join(f"`transforms.{stage}`" for stage in missing)
    raise ValueError(
        f"Nothing prepares the pixels of a split this run reads: {named} "
        f"{'declares' if len(missing) == 1 else 'declare'} no chain, and {', '.join(moved)} "
        f"{'moves' if len(moved) == 1 else 'move'} with the picture. Declare the chain — one ends with "
        "Resize, Normalize and ToTensorV2 (configs/transforms) — or leave the split out of the run."
    )


def _refuse_a_chain_that_does_not_scale_as_the_declaration_promises(
    info: DatasetInfo, transforms: Mapping[str, SampleTransform]
) -> None:
    """What an image becomes is declared in one section and done by another, and nothing compared them.

    `preprocessing.inputs.<name>` is what the record beside an exported artifact promises whoever serves
    it, and what a sample page un-scales pixels by; the arithmetic is the chain a stage declares, which
    is free to do something else entirely. Measured: mean and std of 0.5 beside a `Normalize` told the
    pixels already run 0..1 leaves a white pixel at 509 where the declaration puts it at 1 — and both
    halves are self-consistent, so no number of the run looks wrong.

    The chain is *asked* rather than read: searching it for a `Normalize` to compare parameters against
    would be a second reading of somebody else's operation, defeated by any other spelling of the same
    arithmetic. Only the stages that are not training, because an augmentation draws — what one does to
    a pixel is not a fact about the run — and because what the promise is about is inference, which the
    evaluation chains are the shape of.
    """
    for name, declared in info.inputs.items():
        shape = declared.shape
        if declared.normalization is None or not isinstance(shape, TensorShape):
            continue
        height, width = shape.size(Axis.HEIGHT), shape.size(Axis.WIDTH)
        if height is None or width is None:
            continue
        for stage, chain in transforms.items():
            if stage != Stage.TRAIN:
                _refuse_one_chain(name, stage, chain, declared.normalization, (height, width))


def _refuse_one_chain(
    name: str, stage: str, chain: SampleTransform, declared: Normalization, size: tuple[int, int]
) -> None:
    """One known image through one chain, against where the declaration says each channel should land.

    A different level per channel rather than one flat grey, so that a chain putting the channels in
    another order than the statistics were written for is caught by the same probe as a wrong scale.

    Read from the trailing axes, because what a chain answers with is not always one image: a stage
    drawing views answers with a stack of them, and every draw is scaled the same way, so every draw is
    checked.
    """
    levels = np.linspace(SHADES[0], SHADES[1], len(declared.mean)).round()
    probe = np.broadcast_to(levels.astype(np.uint8), (*size, len(levels)))
    answered = require_tensor(chain(Sample(inputs={name: np.squeeze(probe)})).inputs[name], name=name)
    wanted = [
        round((float(level) / FULL - mean) / deviation, 4)
        for level, mean, deviation in zip(levels, declared.mean, declared.std, strict=True)
    ]
    for drawn in answered[..., 0, 0].reshape(-1, len(levels)).tolist():
        landed = [round(one, 4) for one in drawn]
        if any(abs(one - other) > TOLERANCE for one, other in zip(landed, wanted, strict=True)):
            raise ValueError(
                f"The chain `transforms.{stage}` does not scale {name!r} as `preprocessing.inputs` "
                f"declares: a pixel reading {levels.tolist()} out of {FULL} arrives as {landed}, where "
                f"mean {list(declared.mean)} and std {list(declared.std)} put it at {wanted}. That "
                f"declaration is what the record beside an exported artifact promises whoever serves "
                f"the model, so the two have to be one preprocessing. A `Normalize` told the pixels "
                f"already run 0..1 is the usual cause; channels in another order than the statistics "
                f"were written for is the other."
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
    with naming(f"tasks.{task.name}.loss"):
        return build_loss(declared.loss if declared.loss is not None else task.default_loss, task.facts())


def _refuse_a_head_and_an_objective_that_disagree(model: Model, losses: Mapping[str, Loss]) -> None:
    """Every task's own objective against the head serving it; what the pair has to satisfy is stated once.

    Here because only the root holds both — the network and the objectives are built from separate
    sections — while the sentence those two have to satisfy lives beside the losses, where the learner's
    own term asks it as well.
    """
    for name, loss in losses.items():
        refuse_an_objective_the_head_does_not_answer(name, model.produces(name), loss, f"tasks.{name}.loss")


def _refuse_a_head_of_several_streams_under_an_objective_that_reads_one(
    heads: Mapping[str, HeadConfig], losses: Mapping[str, Loss]
) -> None:
    """A head reading several streams answers once per stream for every sample; most objectives read one.

    Here because only the root holds both: how many streams a head was declared over belongs to the
    model section, and how many answers an objective compares belongs to the objective. Left to meet on
    the first batch, the pair died inside torch with `Expected input batch_size (4) to match target
    batch_size (2)`, naming neither declaration nor which of the two to change.

    One stream under an objective reading two is deliberately not refused: a stage drawing views supplies
    the second answer, and how many it drew is known to the batch alone. Only the direction that can
    never work is refused here.
    """
    for name, head in heads.items():
        streams, reads = len(head.streams), losses[name].reads_per_sample
        if streams > 1 and streams != reads:
            raise ValueError(
                f"Task {name!r}: `head.stream` names {streams} features, so this head answers {streams} "
                f"times for every sample, and objective {losses[name].log_name!r} reads {reads}. Name one "
                f"stream, or declare an objective that learns from several answers of a sample — "
                f"`info_nce` reads a pair."
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

    Read off the declared key rather than off the built saver, and that is not the compromise the three
    pairings above make: ``monitor`` is an argument, so every spelling of the class writes it the same
    way, while *which class* a declaration named is the thing a declaration cannot be asked.

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
    with naming(f"tasks.{task.name}.metrics"):
        return build_metrics(declared.metrics if declared.metrics is not None else task.default_metrics, task.facts())


def build_trainer(config: ExperimentConfig, tracker: Logger | None, callbacks: Sequence[Callback]) -> L.Trainer:
    """The loop itself: how long it runs, what it records to, and what runs alongside it.

    Handed the two rather than building them, because both were built before the data was read: what a
    run records to and what runs alongside it are also what three refusals above are about, and building
    them twice would leave the run holding different objects than the ones that were answered for.

    ``logger=False`` rather than None where a run declares no tracker: left to itself Lightning starts
    a logger of its own, which is not what `tracker: none` says. Where a run's files land is written in
    config as ``${run.directory}``, because Lightning would otherwise resolve it from the tracker.
    """
    return L.Trainer(
        max_epochs=config.epochs,
        default_root_dir=config.run.directory,
        logger=tracker if tracker is not None else False,
        callbacks=list(callbacks),
        profiler=build_profiler(config.trainer.profiler),
        **config.trainer.model_dump(exclude={"profiler"}),
    )
