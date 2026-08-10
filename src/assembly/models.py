"""Building the model — the seam where model families differ."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from src.assembly.checkpoints import load_weights
from src.assembly.instantiate import instantiate
from src.assembly.tasks import build_criterion, build_task_entities, build_tasks
from src.assembly.vendor import is_vendor_family
from src.models import CompositeModel, DistilledModel
from src.models.registry import adapter_registry, backbone_registry, vendor_model_registry
from src.training import TrainingModule

if TYPE_CHECKING:
    from src.config import ExperimentConfig
    from src.config.distillation import TeacherConfig
    from src.core.entities import DataProfile, Task
    from src.core.ports import Model
    from src.models.adapters import Adapters


def backbone_path(config: ExperimentConfig) -> str:
    """Where the backbone sits in the training module — the dot-path a freeze callback names.

    Built by joining the segment each owner publishes, and derived rather than
    fixed: distillation nests the student, so the path follows what this run
    actually builds. A constant would still read ``model.backbone`` in a distilled
    run, where nothing is at that address — and the guard below, which matches
    config against it, would then quietly match nothing.
    """
    nesting = () if config.distillation is None else (DistilledModel.STUDENT,)
    return ".".join((TrainingModule.MODEL, *nesting, CompositeModel.BACKBONE))


def build_model(config: ExperimentConfig, profile: DataProfile) -> tuple[Model, list[Task]]:
    """Build the model and its tasks.

    This is the family seam, and both families now come through it. ``config.model``
    naming an entry of ``vendor_model_registry`` is a *vendor* family — one that owns its head,
    its loss and its decoding — and it takes the short path: the task entities, and the
    model built with the class count the data profiled. Naming a backbone instead is the
    composite family, and the sequence is backbone → adapters → tasks → components →
    ``CompositeModel`` → teachers.

    Nothing outside this function knows which it was, because ``Model`` and ``Task`` are
    all the rest of assembly consumes.

    Adapters go on before the tasks so the heads that follow are untouched by the
    freezing, and before any weights are read — which is the whole reason this
    cannot be a fit-time callback: a checkpoint from an adapted run is keyed for a
    model that already has them.
    """
    if is_vendor_family(config):
        return _vendor_model(config, profile), build_task_entities(config, profile)
    _refuse_a_name_from_neither_registry(config)
    backbone = instantiate(config.model, backbone_registry)
    if config.adapters is not None:
        _refuse_a_second_owner_of_the_backbone(config)
        adapters: Adapters = instantiate(config.adapters, adapter_registry)
        adapters(backbone)
    tasks, components = build_tasks(config, profile, backbone)
    model: Model = CompositeModel(backbone=backbone, components=components)
    if config.distillation is not None:
        model = DistilledModel(
            student=model,
            teachers=[_teacher(one, config, profile) for one in config.distillation.teachers],
            # No derived facts: the comparison spans every task, so there is no one task's
            # sizing to offer it — and comparing two logit tensors needs none.
            criterion=build_criterion(config.distillation.loss),
        )
    return model, tasks


def _teacher(declared: TeacherConfig, config: ExperimentConfig, profile: DataProfile) -> Model:
    """This run's tasks on another backbone — so the two models' logits match by construction.

    Deliberately not ``build_model`` on an altered copy of the config: the list of
    sections a teacher must not inherit would then live in a data structure, and
    the next section added at the root would be inherited in silence. Building the
    two pieces a teacher actually needs says the same thing and cannot go stale.
    ``build_tasks`` only reads the profile, so the second call is safe.

    It also builds criteria and metric containers the teacher will never use — a
    teacher is only ever asked for logits. That is deliberate: sizing a head from
    the run's tasks is the part that must not be duplicated, and a flag to skip
    the rest would be a parameter standing in for a decision, over a few unused
    objects per run.
    """
    backbone = instantiate(declared.backbone, backbone_registry)
    _, components = build_tasks(config, profile, backbone)
    teacher = CompositeModel(backbone=backbone, components=components)
    if declared.checkpoint_path is not None:
        load_weights(teacher, declared.checkpoint_path)
    return teacher


def _refuse_a_name_from_neither_registry(config: ExperimentConfig) -> None:
    """Name both groups when the model section names neither, and say how they differ.

    One key chooses between two registries, so a misspelling falls through to the
    backbone one and is answered with a list of backbones. Measured: ``name: yolov8``
    yields *"Unknown backbone 'yolov8'. Registered: timm, smp, hf, ..."* — a list that
    cannot contain what the user was reaching for, and no hint that a second group
    exists. The framework's rule for a refusal is to list the valid options; here it
    listed half of them.
    """
    name = config.model.name
    if name is None or name in backbone_registry or name in vendor_model_registry:
        return
    composed = ", ".join(sorted(str(key) for key in backbone_registry))
    whole = ", ".join(sorted(str(key) for key in vendor_model_registry))
    raise LookupError(
        f"Unknown model '{name}'. The 'model' section takes either a backbone this framework "
        f"composes heads onto ({composed}), or a family that arrives whole and brings its own "
        f"head, loss and decoding ({whole}). Use '_target_' for anything unregistered."
    )


def _refuse_a_second_owner_of_the_backbone(config: ExperimentConfig) -> None:
    """Adapters freeze the backbone themselves; a freeze callback would freeze them too.

    ``Freeze`` works through Lightning's ``BaseFinetuning``, which runs after
    assembly and sets ``requires_grad=False`` across the module it is given — the
    adapters included, since they live inside the backbone. Training would then
    proceed with nothing to learn, and the only symptom would be a loss that does
    not move.
    """
    held = backbone_path(config)
    contested = [
        declared
        for declared in config.callbacks or []
        if declared.name == "freeze"
        and any(str(module).startswith(held) for module in declared.params.get("modules", []))
    ]
    if contested:
        raise ValueError(
            f"Both 'adapters' and a 'freeze' callback claim {held}: the adapters already hold "
            "the base still, and freezing it again would hold the adapters too, leaving nothing to "
            "learn. Drop the freeze callback, or drop the adapters."
        )


def _vendor_model(config: ExperimentConfig, profile: DataProfile) -> Model:
    """A family that arrives whole, sized from the facts the data declared.

    ``num_classes`` is offered the way every derived fact is, so it reaches a family that
    names it and never has to be restated in config. Everything the constructor does not
    name is the vendor's own configuration — its loss gains and its augmentation knobs
    alike — and is told apart from ours by the signature rather than by a hand-kept table,
    which is what lets the same declaration serve the model and the data pipeline.
    """
    (task_name,) = config.tasks
    return cast(
        "Model", instantiate(config.model, vendor_model_registry, num_classes=profile.require_num_classes(task_name))
    )
