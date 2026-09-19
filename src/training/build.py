"""Building what a run trains with: the learner over the assembled parts, and the two optimization factories."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from inspect import signature
from typing import TYPE_CHECKING, Any

from lightning.pytorch.profilers import Profiler
from torch.optim import Optimizer

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import LRSchedulerConfigType

from src.config import ComponentConfig, HeadConfig, LearnerConfig, SchedulerConfig, TeacherConfig
from src.config.instantiate import fill_signature, instantiate, instantiate_offering, resolve_factory, resolve_params
from src.core import TensorShape, naming
from src.losses import Loss
from src.losses.registry import distillation_loss_registry
from src.models import Model, load_weights
from src.models.build import build_model
from src.tasks import Task
from src.training.base import FitProfile, Learner, OptimizerFactory, SchedulerFactory
from src.training.checkpoints import model_weights
from src.training.registry import learner_registry, optimizer_registry, profiler_registry, scheduler_registry

log = logging.getLogger(__name__)

LEARNING_RATE = "lr"
"""What a learning-rate graph is titled; Lightning's monitor reads it from the policy below.

Left unset, the monitor titles the graph after the optimizer's class — ``lr-AdamW/backbone`` — which
puts the reader's comparison under a name they are not comparing. Under ``lr`` the rates of every
group share one graph, each a line on it.
"""


def _names(factory: Callable[..., Any], part: str) -> bool:
    """Whether a constructor asks for one of the parts ``build_learner`` offers, asked the way it offers it.

    The one question behind every refusal about a section that means nothing alone — a teacher with
    nobody to read it, an objective nobody would ask. A registry name would settle it for the shipped
    learners alone: ``_target_`` writes none, and a learner of a reader's own is as much half of a pair
    as these are, so what settles it is the constructor.
    """
    return part in fill_signature(factory, **{part: None})


def learners_naming(part: str) -> str:
    """Every registered learner that part may be declared beside, so a refusal names the fix.

    Read off the registry rather than spelled beside it: a learner registered tomorrow is offered by the
    same sentence that day, and a list written by hand could fall out of step with one.
    """
    return ", ".join(sorted(name for name in learner_registry if _names(learner_registry.get(name), part)))


def reads_a_teacher(declared: ComponentConfig) -> bool:
    """Whether this learner would be handed a teacher, which is what makes a `teacher` section mean something."""
    return _names(resolve_factory(declared, learner_registry), "teacher")


def build_objective(declared: LearnerConfig) -> Loss | None:
    """The objective declared below the learner, built from the registry that serves that position.

    Resolved here rather than left among the learner's own arguments: a registry belongs to a position,
    so a ``name`` one level down resolves to nothing and would travel on as the mapping it literally is.
    This is the same child position ``model.backbone`` is, filled the same way by the builder that owns it.
    """
    if declared.loss is None:
        return None
    with naming("learner.loss"):
        built = instantiate(declared.loss, distillation_loss_registry)
        if not isinstance(built, Loss):
            raise TypeError(
                f"{declared.loss.spelled!r} built {type(built).__name__}, which does not compare this run's "
                "answer with the teacher's: an objective takes both and reports one number."
            )
        return built


def _refuse_an_objective_this_learner_would_never_read(declared: LearnerConfig) -> None:
    """A position filled below a learner that names none is a declaration nothing would ever read.

    An offered fact a constructor does not name is dropped by contract — that is how one learner takes
    ``losses`` and another does not. A *declaration* carries no such contract: it was written meaning to
    take effect, and dropping it in silence is how a run trains by an objective nobody chose while its
    log reads like any other.

    Asked of the constructor rather than of the name, for the reason a teacher is: a ``_target_``
    declaration writes no name, so keying on one would let exactly the learners that most need saying
    so pass unnoticed.
    """
    if declared.loss is None:
        return
    if not _names(resolve_factory(declared, learner_registry), "loss"):
        raise ValueError(
            f"`learner.loss` declares an objective and `learner` is {declared.spelled!r}, which names no "
            f"`loss` in its constructor: it would be built and then asked nothing. Declare a learner that "
            f"reads one — {learners_naming('loss')} — or drop `learner.loss`."
        )


def build_learner(
    declared: LearnerConfig,
    *,
    model: Model,
    tasks: Mapping[str, Task],
    losses: Mapping[str, Loss],
    teacher: Model | None = None,
) -> Learner:
    """The algorithm a run trains by, over the parts it has already assembled.

    Offered rather than imposed: `Learner` asks for a model and its tasks, and an algorithm that owns
    its objective some other way — one whole model carrying its own loss, a distilled pair — implements
    that contract and names no `losses`. Imposing them made every such learner take an argument it had
    no use for, which is the extension point promising one thing and the builder demanding another.

    ``learned_only`` is offered the same way and derived rather than declared: a target whose vocabulary
    the training split settled says so, and an objective keeping one parameter per entry of it has
    nothing to say about an entry no split it learned from held.

    So is ``teacher``, which is why a second network is declared in a section of its own rather than
    inside this one: offered as a fact *and* written in the declaration, it would be two statements of
    one thing, and the builder refuses those by name before either could be read.
    """
    learned_only = sorted(name for name, task in tasks.items() if task.info.open_set)
    _refuse_a_total_that_would_mean_two_things(tasks, learned_only)
    _refuse_an_objective_this_learner_would_never_read(declared)
    objective = build_objective(declared)
    built = instantiate_offering(
        declared,
        learner_registry,
        model=model,
        tasks=tasks,
        losses=losses,
        learned_only=learned_only,
        teacher=teacher,
        loss=objective,
    )
    if not isinstance(built, Learner):
        raise TypeError(
            f"{declared.spelled!r} built {type(built).__name__}, which is not a Learner: it cannot turn a "
            "batch into a loss, and a trainer has nothing to ask it for."
        )
    return built


def build_teacher(
    declared: TeacherConfig | None, *, heads: Mapping[str, HeadConfig], outputs: Mapping[str, TensorShape]
) -> Model | None:
    """The second network a run learns from, sized by this run's tasks and holding the weights it answers with.

    Sized here rather than declared, which is what makes a teacher a teacher rather than a second model:
    it answers the same questions as the student, so its heads are built from the same shapes by the same
    builder. A declaration restating them could disagree with them, and the disagreement would show up as
    a shape error inside a step.

    Its weights are read straight into it, and the strictness that costs is the point twice over: it is
    what makes the teacher worth listening to, and it is also the only check that the file and the
    declaration are about the same network — a teacher built from one architecture and loaded from
    another's run is refused by name rather than by a wrong number.

    Not through ``load_checkpoint``, which says in its log that the optimizer and the epoch counter start
    fresh. True, and about a run continuing from weights; a teacher continues nothing.
    """
    if declared is None:
        return None
    teacher = build_model(declared, heads=heads, outputs=outputs)
    load_weights(teacher, model_weights(declared.checkpoint_path), declared.checkpoint_path)
    log.info("The teacher answers with the weights from %s and learns nothing here.", declared.checkpoint_path)
    return teacher


def _refuse_a_total_that_would_mean_two_things(tasks: Mapping[str, Task], learned_only: Sequence[str]) -> None:
    """A run whose objective stops for one task and not for another totals two different things.

    Here because the set of tasks is what settles it, and this is the builder that reads that set. The
    total is logged under one name in every stage, so a run mixing the two would put a training sum and
    a smaller evaluation sum on one chart with nothing saying which is which — and a checkpoint
    monitoring it would choose on half the objective. The per-task terms stay readable either way; it is
    the total that cannot be made honest without saying, per stage, what it totalled.
    """
    if learned_only and len(learned_only) != len(tasks):
        scored = sorted(set(tasks) - set(learned_only))
        raise ValueError(
            f"{', '.join(learned_only)} is judged on a vocabulary the training split settled, so its "
            f"objective stops outside training, while {', '.join(scored)} is scored in every stage. The "
            f"run's total would mean one thing in training and another in evaluation, under one name. "
            f"Train them as separate runs, or pin the vocabulary with `tasks.<name>.target_encoder: "
            f"label` and `tasks.<name>.classes` so every objective is scored everywhere."
        )


def build_optimizer_factory(declared: ComponentConfig, lr: float) -> OptimizerFactory:
    """A factory, not an optimizer: the parameters it moves do not exist while a config is being read.

    The run's rate is the base every group inherits; a group that declared one of its own keeps it.
    """
    return partial(resolve_factory(declared, optimizer_registry), lr=lr, **resolve_params(declared))


def _reacts_to_a_metric(schedule: Any) -> bool:
    """Whether a schedule steps on a logged number rather than on the clock — asked of the library.

    torch distinguishes the two in the signature it publishes: ``ReduceLROnPlateau.step`` takes the
    metric, every other schedule's takes at most an epoch. Asked that way rather than by naming the one
    class, for the reason ``_derived`` gives below: a table of library names is a thing to keep in step,
    and a schedule reached by ``_target_`` would not be in it.
    """
    step = getattr(schedule, "step", None)
    return step is not None and "metrics" in signature(step).parameters


def build_scheduler_factory(declared: SchedulerConfig | None) -> SchedulerFactory | None:
    """A factory too, one step later: a schedule needs the built optimizer and the length of the fit."""
    if declared is None:
        return None
    schedule = resolve_factory(declared, scheduler_registry)
    if _reacts_to_a_metric(schedule) and declared.monitor is None:
        raise ValueError(
            "A plateau schedule reacts to a logged metric, so it needs 'monitor' — "
            "e.g. scheduler: {name: plateau, monitor: val/loss, mode: min}."
        )

    def factory(optimizer: Optimizer, profile: FitProfile) -> LRSchedulerConfigType:
        written = resolve_params(declared)
        derived = _derived(schedule, optimizer, profile, written)
        _refuse_a_schedule_on_the_wrong_clock(declared.spelled, declared.interval, derived, profile)
        policy: LRSchedulerConfigType = {
            "scheduler": schedule(optimizer, **written, **derived),
            "name": LEARNING_RATE,
            "interval": declared.interval,
            "frequency": declared.frequency,
            "strict": declared.strict,
        }
        if declared.monitor is not None:
            policy["monitor"] = declared.monitor
        return policy

    return factory


def _derived(
    schedule: Callable[..., Any], optimizer: Optimizer, profile: FitProfile, written: Mapping[str, Any]
) -> dict[str, Any]:
    """The facts a schedule names and the run alone knows: how long the fit is, and each group's rate.

    Only torch's canonical spellings are filled, so no table of per-scheduler names has to be kept in
    step with the library. Which clock a schedule runs on follows from what it names — ``total_steps``
    where it has one, the epoch pair otherwise — and never from both, which torch refuses outright.
    A run that wrote one of these itself keeps it: shaping a schedule deliberately is a real recipe.
    """
    clock = fill_signature(schedule, total_steps=profile.total_steps) or fill_signature(
        schedule, steps_per_epoch=profile.steps_per_epoch, epochs=profile.epochs
    )
    # A schedule that sets rates outright takes one per group: a single number would broadcast over
    # every group and quietly undo the rate a task declared for itself.
    peaks = fill_signature(schedule, max_lr=[group["lr"] for group in optimizer.param_groups])
    return {name: value for name, value in {**clock, **peaks}.items() if name not in written}


def _refuse_a_schedule_on_the_wrong_clock(
    spelled: str, interval: str, derived: Mapping[str, Any], profile: FitProfile
) -> None:
    stepwise = sorted({"total_steps", "steps_per_epoch"} & derived.keys())
    if stepwise and interval == "epoch":
        raise ValueError(
            f"{spelled!r} was sized in optimizer steps ({', '.join(stepwise)} filled from the fit), but its "
            f"interval is 'epoch': it would advance {profile.epochs} of {profile.total_steps} steps and hold "
            "its warm-up rate for the whole run. Set interval: step."
        )


def build_profiler(declared: ComponentConfig | None) -> Profiler | None:
    """Where a run's wall clock went, when a run asks — `trainer=profile` is the shipped way to."""
    if declared is None:
        return None
    built = instantiate(declared, profiler_registry)
    if not isinstance(built, Profiler):
        raise TypeError(f"{declared.spelled!r} built {type(built).__name__}, which profiles nothing.")
    return built
