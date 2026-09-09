"""The composition root: a validated experiment declaration becomes a running experiment.

Only wiring lives here — the one place that reads a whole ``ExperimentConfig`` and hands
each package the section it builds from (``docs/adr/0001-one-composition-root.md``). How a
thing is *made* — a dataset read, a criterion composed, a metric set sized, an optimizer's
parameter groups, weights put into a model, an artifact written and verified — belongs to that
package. What stays here is the wiring: the order things are made in, the facts that travel
between them, and the declarations only a whole config can answer — which checkpoint prefix to
strip, what shape an export example takes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightning as L
import torch
from lightning import seed_everything

from src.callbacks.registry import callback_registry
from src.config.components import MetricConfig
from src.config.instantiate import instantiate, refuse_a_declared_fact, resolve_params, resolve_target
from src.core.entities import TaskFacts
from src.core.ports import Backbone, Model
from src.core.taxonomy import Geometry
from src.data.build import build_pipeline, input_geometries
from src.export import DeployableModel
from src.export.build import build_exporters
from src.export.ship import ExampleRejected, ship
from src.loggers import logger_registry
from src.loggers.ports import TagsRuns
from src.losses.build import build_criterion
from src.metrics.build import build_metric_sets
from src.models import CompositeModel, DistilledModel, load_weights, merge_adapters, without_teachers
from src.models.registry import adapter_registry, backbone_registry, head_registry
from src.tasks import Overrides, Task
from src.tasks.entities import NATIVE_HEAD
from src.tasks.registry import task_kind_registry
from src.training import TrainingData, TrainingModule
from src.training.build import build_optimizer_factory, build_scheduler_factory
from src.training.registry import profiler_registry

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from lightning.pytorch.callbacks import Callback
    from torch import Tensor, nn

    from src.config import ExperimentConfig
    from src.config.distillation import TeacherConfig
    from src.config.tasks import HeadConfig, TaskConfig
    from src.core.entities import DatasetFacts
    from src.core.taxonomy import Stage
    from src.data.datamodules.base import DataModule
    from src.export import ExportedArtifact, Exporter
    from src.metrics.ports import MetricSet
    from src.models import TaskComponents
    from src.models.adapters import Adapters
    from src.tasks import TaskKind
    from src.tasks.entities import NativeHead

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Experiment:
    """Everything a run needs, built and ready."""

    module: TrainingModule
    data: TrainingData
    trainer: L.Trainer
    exporters: list[Exporter]


def build(config: ExperimentConfig) -> Experiment:
    """Build an experiment from config, in the one order the contract allows.

    ``DataModule.setup`` runs before the model is built: that ordering is what lets output
    sizes come from data instead of config, and it holds for a composed model and for one
    that arrives whole alike, which is why it stays visible here.
    """
    seed_everything(config.seed, workers=True)
    kinds = build_kinds(config)
    data_module = build_data_module(config, kinds)
    facts = data_module.setup()
    model, tasks = build_model(config, facts, kinds)
    _refuse_what_cannot_train_yet(model, tasks)
    module = TrainingModule(
        model=model,
        tasks=tasks,
        metrics=build_metrics(config, facts, kinds),
        optimizer_factory=build_optimizer_factory(config.optimizer),
        scheduler_factory=build_scheduler_factory(config.scheduler),
    )
    return Experiment(
        module=module,
        data=build_training_data(config, data_module),
        trainer=build_trainer(config, architecture=model.architecture),
        exporters=build_exporters(config.export),
    )


def run(experiment: Experiment, config: ExperimentConfig) -> None:
    """Fit, evaluate and ship, as the run section asks.

    Only ``fit`` is ever handed a checkpoint, and only to continue an interrupted run;
    everything after it reads the module, so the weights evaluated and shipped are the ones
    the run stopped on (measured: handing the same file to ``test`` reloads it). ``shipped``
    excludes a distilled run's frozen teachers.
    """
    # Before anything can fail: what a tracker shows about a run should not depend on
    # the run finishing. Lightning never calls this itself — nothing here saves
    # hyperparameters onto the module, because a module holds built objects, not config.
    if experiment.trainer.logger is not None:
        experiment.trainer.logger.log_hyperparams(config.model_dump(mode="json"))
    shipped = without_teachers(experiment.module.model)
    if config.run.checkpoint_path is not None:
        load_checkpoint(shipped, config.run.checkpoint_path)
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
    load_checkpoint(model, path)


def _refuse_what_cannot_train_yet(model: Model, tasks: Sequence[Task]) -> None:
    """A composed model over a kind that has no criterion yet builds, and must not train.

    Stage scaffolding for detection: the composed model builds — stage 2's promise, which
    its tests hold — and nothing trains it until its criterion lands (roadmap stages 3–4). Asked of the
    built model rather than of the config, because a model that arrives whole serves the
    same kind with a criterion of its own. Goes with that criterion.
    """
    if not isinstance(without_teachers(model), CompositeModel):
        return
    for task in tasks:
        if task.kind.unavailable is not None:
            raise ValueError(f"Task '{task.name}' is '{type(task.kind).__name__}': {task.kind.unavailable}")


def build_kinds(config: ExperimentConfig) -> dict[str, TaskKind]:
    """Every task's kind, built once here and handed to each package that asks.

    The data pipeline chooses encoders by it, the tasks carry it, the metrics read its
    defaults: three readers, one object per task, so a kind with a side effect in its
    constructor sees the run once and nothing holds a kind another reader did not get.
    """
    return {name: kind_of(declared) for name, declared in config.tasks.items()}


def build_data_module(config: ExperimentConfig, kinds: Mapping[str, TaskKind]) -> DataModule:
    """The table pipeline, over the kinds its target encoders and transforms are chosen by."""
    return build_pipeline(config.data, config.tasks, kinds, config.transforms)


def kind_of(declared: TaskConfig) -> TaskKind:
    """The kind a task declares — a registered name or an import path, argument-free by contract."""
    built: TaskKind = instantiate(declared.kind, task_kind_registry)
    return built


def build_task_entities(config: ExperimentConfig, facts: DatasetFacts, kinds: Mapping[str, TaskKind]) -> list[Task]:
    """The declared tasks as entities, each with the facts the data revealed about it.

    Split from ``build_tasks`` because a model that arrives whole needs exactly this half: it binds its
    own head and criterion. Runs after ``DataModule.setup`` — the facts come from it.
    """
    return [
        Task(
            name=name,
            kind=kinds[name],
            facts=facts.get(name, TaskFacts()),
            weight=declared.weight,
            lr=declared.lr,
        )
        for name, declared in config.tasks.items()
    ]


def build_tasks(
    config: ExperimentConfig, facts: DatasetFacts, backbone: Backbone, kinds: Mapping[str, TaskKind]
) -> tuple[list[Task], dict[str, TaskComponents]]:
    """Build the declared tasks and the composite components serving each.

    Runs after ``DataModule.setup``: head widths come from the facts it returned, never from
    config. What the declaration overrides — head, streams, loss — reaches the kind as
    ``Overrides``; everything else is the kind's own answer.
    """
    tasks = build_task_entities(config, facts, kinds)
    components = {
        task.name: task.kind.components(task, backbone, _overrides_of(config.tasks[task.name])) for task in tasks
    }
    return tasks, components


def _overrides_of(declared: TaskConfig) -> Overrides:
    """The declaration's overrides as factories: the kind resolves the sizes they need."""
    return Overrides(
        head=_head_override(declared.head) if declared.head is not None else None,
        streams=declared.streams,
        loss=partial(build_criterion, declared.loss) if declared.loss is not None else None,
    )


def _head_override(declared: HeadConfig) -> NativeHead | Callable[[int | tuple[int, ...], int], nn.Module]:
    """The declared head as the kind resolves it: the backbone's own by its reserved name, or a factory to size."""
    if declared.name != NATIVE_HEAD:
        return partial(_build_head, declared)
    if declared.params:
        raise ValueError(
            f"'native' takes no arguments: it names the head the backbone brings, built by the backbone. "
            f"Drop {', '.join(declared.params)} from the head."
        )
    return NATIVE_HEAD


def _build_head(declared: HeadConfig, in_features: int | tuple[int, ...], out_features: int) -> nn.Module:
    """The declared kind of head, at the sizes the kind resolved.

    Every head takes ``in_features`` and ``out_features`` — the one contract a head has — so
    ``{name: cosine}`` is a complete declaration and config never repeats what the
    backbone and the data already said; a size written there anyway is refused by name.
    """
    refuse_a_declared_fact(declared, "in_features", "out_features")
    factory = resolve_target(declared, head_registry)
    built: nn.Module = factory(in_features=in_features, out_features=out_features, **resolve_params(declared))
    return built


def build_model(
    config: ExperimentConfig, facts: DatasetFacts, kinds: Mapping[str, TaskKind]
) -> tuple[Model, list[Task]]:
    """Build the model and its tasks.

    The model section builds one of two things. A ``Model`` — reached by ``_target_``, owning
    its head, loss and decoding — is taken as it is; the sections that reparameterize or
    compare a composed model refuse it by name. A ``Backbone`` is the composite family:
    backbone → adapters → tasks → components → ``CompositeModel`` → teachers. Adapters go on
    before the tasks and before any weights are read: a checkpoint from an adapted run is
    keyed for a model that already has them.
    """
    built = instantiate(config.model, backbone_registry)
    if isinstance(built, Model):
        _refuse_what_a_whole_model_cannot_serve(config, built)
        return built, build_task_entities(config, facts, kinds)
    if not isinstance(built, Backbone):
        raise TypeError(
            f"The 'model' section built {type(built).__name__}, which is neither a Backbone this framework "
            f"composes heads onto nor a Model that arrives whole."
        )
    backbone: Backbone = built
    if config.adapters is not None:
        _refuse_a_second_owner_of_the_backbone(config)
        adapters: Adapters = instantiate(config.adapters, adapter_registry)
        adapters(backbone)
    tasks, components = build_tasks(config, facts, backbone, kinds)
    model: Model = CompositeModel(backbone=backbone, components=components)
    if config.distillation is not None:
        model = DistilledModel(
            student=model,
            teachers=[_teacher(one, config, facts, kinds) for one in config.distillation.teachers],
            # No derived facts: the comparison spans every task, so there is no one task's
            # sizing to offer it — and comparing two logit tensors needs none.
            criterion=build_criterion(config.distillation.loss),
        )
    return model, tasks


def _refuse_what_a_whole_model_cannot_serve(config: ExperimentConfig, model: Model) -> None:
    """Adapters reparameterize a composed backbone; distillation compares composed logits.

    A model that arrives whole offers neither, and a section silently ignored is worse than
    a run that dies: it would report numbers for a recipe nobody ran.
    """
    declared = [section for section in ("adapters", "distillation") if getattr(config, section) is not None]
    if declared:
        raise ValueError(
            f"'{declared[0]}' needs a model this framework composed, and {type(model).__name__} arrives whole. "
            f"Drop the section, or declare a backbone to compose."
        )


def _teacher(
    declared: TeacherConfig, config: ExperimentConfig, facts: DatasetFacts, kinds: Mapping[str, TaskKind]
) -> Model:
    """This run's tasks on another backbone — so the two models' logits match by construction.

    Deliberately not ``build_model`` on an altered copy of the config: the sections a
    teacher must not inherit would then be a list that goes stale. It also builds criteria
    the teacher never uses; sizing the heads from the run's tasks is the part that must
    not be duplicated.
    """
    backbone = instantiate(declared.backbone, backbone_registry)
    _, components = build_tasks(config, facts, backbone, kinds)
    teacher = CompositeModel(backbone=backbone, components=components)
    if declared.checkpoint_path is not None:
        load_checkpoint(teacher, declared.checkpoint_path)
    return teacher


def _refuse_a_second_owner_of_the_backbone(config: ExperimentConfig) -> None:
    """Adapters freeze the backbone themselves; a freeze callback would freeze them too.

    ``Freeze`` works through Lightning's ``BaseFinetuning``, which runs at fit start
    and sets ``requires_grad=False`` across the module it is given — the
    adapters included, since they live inside the backbone. Training would then
    proceed with nothing to learn, and the only symptom would be a loss that does
    not move.
    """
    held = CompositeModel.BACKBONE
    contested = [
        declared
        for declared in config.callbacks or []
        if declared.name == "freeze"
        and any(
            str(module) == held or str(module).startswith(f"{held}.") for module in declared.params.get("modules", [])
        )
    ]
    if contested:
        raise ValueError(
            f"Both 'adapters' and a 'freeze' callback claim '{held}': the adapters already hold "
            "the base still, and freezing it again would hold the adapters too, leaving nothing to "
            "learn. Drop the freeze callback, or drop the adapters."
        )


def build_metrics(
    config: ExperimentConfig, facts: DatasetFacts, kinds: Mapping[str, TaskKind]
) -> dict[str, dict[Stage, MetricSet]]:
    """Every task's metric sets, per stage — what the training module registers and reports.

    A task that declares no ``metrics`` is judged by its kind's default set, resolved here
    rather than when the config loads, because the kind is a fact of ``tasks/`` and config
    reads only ``core``; the substitution is said out loud. A declared mapping replaces the
    default whole.
    """
    built: dict[str, dict[Stage, MetricSet]] = {}
    for name, declared in config.tasks.items():
        kind = kinds[name]
        if declared.metrics is None:
            metrics = {label: MetricConfig.model_validate(dict(spec)) for label, spec in kind.default_metrics.items()}
            if metrics:
                log.info("Task '%s': metrics from kind '%s': %s.", name, declared.kind.spelled, ", ".join(metrics))
        else:
            metrics = dict(declared.metrics)
        built[name] = build_metric_sets(kind, facts.get(name, TaskFacts()), metrics)
    return built


def build_training_data(config: ExperimentConfig, data_module: DataModule) -> TrainingData:
    """Loader knobs forward verbatim, including ones the adapter never declared."""
    return TrainingData(data_module, **config.loader.model_dump())


def build_trainer(config: ExperimentConfig, architecture: str | None = None) -> L.Trainer:
    """The trainer, rooted at the run's output directory and carrying its callbacks.

    Where a run's own files land is written in config as ``${run.directory}/...``, not
    decided here: Lightning would otherwise resolve them from the logger, which is wrong
    for a tracker that uploads. ``architecture`` tags the run afterwards, through the
    ``TagsRuns`` port — config cannot supply it, because the key naming an architecture
    differs per family, and a logger without the port is left alone.
    """
    tracker: dict[str, Any] = {}
    if config.logger is not None:
        logger = instantiate(config.logger, logger_registry)
        if isinstance(logger, TagsRuns):
            logger.tag_run(architecture)
        tracker["logger"] = logger
    profiler = None if config.trainer.profiler is None else instantiate(config.trainer.profiler, profiler_registry)
    return L.Trainer(
        **config.trainer.model_dump(exclude={"profiler"}),
        callbacks=build_callbacks(config),
        profiler=profiler,
        **tracker,
    )


def build_callbacks(config: ExperimentConfig) -> list[Callback]:
    """The declared callbacks, in the order the file gives them.

    One grammar builds them, from their declarations alone. A callback that needs the run's
    tasks reads them off the module in ``setup``, the way any Lightning callback reads the
    module — nothing is offered here by name. A shared config value (``mean``, ``lr``) is
    reached by interpolation, ``${mean}``, as the transforms reach it.
    """
    if config.callbacks is None:
        return []
    built: list[Callback] = [instantiate(declared, callback_registry) for declared in config.callbacks]
    return built


def shipped_weights(path: str) -> dict[str, Tensor]:
    """The weights a run's checkpoint holds for the model that ships.

    A run writes its whole training module, and a distilled run nests the student one level
    further. Unwrapping here gives one rule for every file this framework writes: a
    checkpoint carries the shipped model's weights, and a run's scaffolding (teachers, a
    criterion's state) is not part of them, so one file loads into a distilled and a plain
    model alike. ``weights_only=True`` is enough for a Lightning checkpoint (measured). A
    file without a ``state_dict`` is refused by name: a backbone's own checkpoint
    belongs in ``model.checkpoint_path``.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "state_dict" not in state:
        raise ValueError(
            f"{path} is not a checkpoint this framework wrote: it carries no 'state_dict'. "
            "Arrived weights of a backbone architecture go in 'model.checkpoint_path' instead."
        )
    model_weights = _under(state["state_dict"], TrainingModule.MODEL)
    return _under(model_weights, DistilledModel.STUDENT) or model_weights


def _under(weights: dict[str, Tensor], owner: str) -> dict[str, Tensor]:
    """The entries an attribute contributed, under the names that attribute knows them by."""
    prefix = f"{owner}."
    return {name.removeprefix(prefix): value for name, value in weights.items() if name.startswith(prefix)}


def load_checkpoint(model: nn.Module, path: str) -> None:
    """Put a checkpoint's weights into the model, and nothing else of it.

    Takes the model rather than the training module: what a checkpoint is *about* is the
    model, and the optimizer and the epoch counter deliberately start fresh — a resumed
    run goes through ``run.resume_path`` and Lightning instead.
    """
    load_weights(model, shipped_weights(path), path)
    log.info("Loaded the weights from %s; the optimizer and the epoch counter start fresh.", path)


def example_inputs(config: ExperimentConfig, batch_size: int) -> tuple[Tensor, ...]:
    """One dummy tensor per declared model input, shaped the way the run feeds them.

    Not a guess: ``image_size`` and ``mean`` are the very fields the shipped transform
    groups hand to ``Resize`` and ``Normalize``, so the example agrees with the run for as long
    as the transforms read them — and export needs no dataset, which is what lets it run from a
    checkpoint anywhere. It is a convention, not a guarantee: a transforms group that resizes to
    something of its own leaves this shape wrong, and the model refusing the example is what
    catches it, named by ``export_model`` against the config fields it came from.
    """
    geometries = input_geometries(config.data.inputs)
    not_pictures = [f"'{name}' ({geometry})" for name, geometry in geometries.items() if geometry is not Geometry.IMAGE]
    if not_pictures:
        raise ValueError(
            f"Export shapes its example from 'image_size' and 'mean', which describe a picture, and input "
            f"{', '.join(not_pictures)} is not one. A model over such an input cannot be exported from a "
            f"checkpoint alone yet."
        )
    channels = len(config.mean)
    height, width = config.image_size
    return tuple(torch.randn(batch_size, channels, height, width) for _ in geometries)


def export_model(model: Model, config: ExperimentConfig, exporters: Sequence[Exporter]) -> list[ExportedArtifact]:
    """Ship the model in every declared format, from the graph the run's declarations shape.

    Raises:
        ValueError: If the example shape is not what the model takes — naming the config fields it came from.
        RuntimeError: If any written artifact drifted outside its tolerance.
    """
    graph = DeployableModel(model, list(config.data.inputs), list(config.tasks))
    destination = Path(config.run.directory or ".") / "export" / "model"
    try:
        return ship(graph, partial(example_inputs, config), exporters, destination)
    except ExampleRejected as rejected:
        raise ValueError(
            f"{rejected} The shape comes from 'image_size' {tuple(config.image_size)} and the "
            f"{len(config.mean)} normalisation channel(s) of 'mean'; an input that is not an image of "
            "that shape cannot be exported yet."
        ) from rejected
