"""Building what a run trains with: the learner over the assembled parts, and the two optimization factories."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from inspect import signature
from typing import TYPE_CHECKING, Any

from lightning.pytorch.profilers import Profiler
from torch.optim import Optimizer

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import LRSchedulerConfigType

from src.config import ComponentConfig, DistilledLossConfig, HeadConfig, LearnerConfig, SchedulerConfig, TeacherConfig
from src.config.instantiate import fill_signature, instantiate, instantiate_offering, resolve_factory, resolve_params
from src.core import TensorShape, naming
from src.losses import Loss
from src.losses.build import build_loss
from src.losses.registry import distillation_loss_registry
from src.models import Model, load_weights
from src.models.build import build_model
from src.tasks import Task
from src.training.base import FitProfile, Learner, OptimizerFactory, SchedulerFactory
from src.training.checkpoints import model_weights
from src.training.distillation import DISTILLATION, REPRESENTATION
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

    The one question behind every refusal about a child position that means nothing alone — a teacher
    with nobody to read it, an objective nobody would ask. A registry name would settle it for the shipped
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


def _declared_terms(declared: LearnerConfig) -> list[DistilledLossConfig]:
    """One grammar out of the two a declaration may use: one objective, or a weighted list of them.

    Shaped like ``losses.build._weighted`` and for its reason: a run writing one term and a run writing
    three arrive here and leave alike, so nothing downstream has to know which way it was written.
    """
    if declared.loss is None:
        return []
    if isinstance(declared.loss, ComponentConfig):
        return [DistilledLossConfig(loss=declared.loss)]
    return list(declared.loss)


def build_objective(declared: LearnerConfig) -> Loss | None:
    """How far this run's answers are from the teacher's: every term below the learner that named no stream.

    Resolved here rather than left among the learner's own arguments: a registry belongs to a position,
    so a ``name`` one level down resolves to nothing and would travel on as the mapping it literally is.
    This is the same child position ``model.backbone`` is, filled the same way by the builder that owns it.
    """
    answered = [term for term in _declared_terms(declared) if not term.streams]
    return _against_the_teacher(answered, DISTILLATION) if answered else None


def build_representation(declared: LearnerConfig) -> Mapping[str, Loss]:
    """How far this run's features are from the teacher's, one objective per stream a term named.

    Per stream rather than one objective over all of them, because the streams are compared separately
    and reported separately: a run pulling ``encoder`` gently and ``decoder`` hard writes two terms, and
    a single sum of them could be neither read back nor weighted apart.

    A separate objective per stream even where one term named several, because a loss is a module: one
    object standing under two names would add their states together the day an objective keeps any.
    """
    over: dict[str, list[DistilledLossConfig]] = defaultdict(list)
    for term in _declared_terms(declared):
        for stream in term.streams:
            over[stream].append(term)
    return {stream: _against_the_teacher(terms, REPRESENTATION) for stream, terms in over.items()}


def _against_the_teacher(terms: Sequence[DistilledLossConfig], reported_as: str) -> Loss:
    """The declared terms as one objective, each reporting under the name its reading gives it.

    Built through the one function that turns a declaration into an objective, so that the normalisation,
    the collapse of a single unweighted term and the fallback to a module reached by ``_target_`` are
    stated once and serve both positions that write one.

    A term's default name comes from what it reads rather than from the loss it uses, so that a column
    survives a change of measure — the reason ``DISTILLATION`` is named for the method rather than for
    the divergence it happens to use. A term that wrote ``log_name`` keeps it, which is how a run tells
    two terms over one reading apart.
    """
    named = [one if one.log_name is not None else one.model_copy(update={"log_name": reported_as}) for one in terms]
    with naming("learner.loss"):
        return build_loss(named, facts={}, registry=distillation_loss_registry)


def refuse_a_learner_and_its_child_positions_that_disagree(declared: LearnerConfig) -> None:
    """What is written under a learner has to be what that learner reads, both ways round.

    One question asked twice — does this algorithm name this part — because both answers are otherwise
    silent. A position nothing reads is built, and a teacher read from its file besides, then asked
    nothing for the whole run under a log that looks like any other's. A position an algorithm needs and
    nobody wrote fails where the learner is assembled, in the words of a missing argument rather than
    naming what to write.

    An offered fact a constructor does not name is dropped by contract — that is how one learner takes
    ``losses`` and another does not. A *declaration* carries no such contract: it was written meaning to
    take effect, so dropping it in silence is the failure this replaces.

    Asked of the constructor rather than of the name a declaration wrote: a ``_target_`` declaration
    writes no name, so keying on one would pass exactly the learners a reader wrote themselves, and one
    of those is as much half of a pair as a shipped one.

    Only ``teacher`` is asked for in the second direction, because it is the only one a run cannot do
    without: an algorithm left with no objective makes the one it says it defaults to, and a network to
    learn from cannot be derived from anything a run already holds.
    """
    factory = resolve_factory(declared, learner_registry)
    for part, written, what in (
        ("loss", declared.loss, "an objective to measure the distance to a second network by"),
        ("teacher", declared.teacher, "a second network to learn from"),
    ):
        if written is not None and not _names(factory, part):
            raise ValueError(
                f"`learner.{part}` declares {what}, and `learner` is {declared.spelled!r}, which names no "
                f"`{part}` in its constructor: it would be built and then asked nothing. Declare a learner "
                f"that reads one — {learners_naming(part)} — or drop `learner.{part}`."
            )
    if any(term.streams for term in _declared_terms(declared)) and not _names(factory, "representation"):
        raise ValueError(
            f"`learner.loss` declares a term over a feature stream, and `learner` is {declared.spelled!r}, "
            f"which names no `representation` in its constructor: the term would be built and then asked "
            f"nothing. Declare a learner that reads one — {learners_naming('representation')} — or drop "
            f"the `stream` from that term, and it compares the answers instead."
        )
    if declared.teacher is None and _names(factory, "teacher"):
        raise ValueError(
            f"`learner` is {declared.spelled!r}, which learns from a second network, and there is nothing "
            f"to distil from. Declare `learner.teacher` — the model it is, and `checkpoint_path` for the "
            f"run whose weights it answers with — or a learner that learns from the data alone."
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

    ``teacher`` and ``loss`` arrive offered too, and are the two an algorithm may be *declared* with:
    each is written as a child position under the learner, resolved by the builder that owns it and then
    handed over as a fact. A typed position never reaches the constructor's own arguments, which is what
    keeps one declaration from being two statements of one thing.
    """
    learned_only = sorted(name for name, task in tasks.items() if task.info.open_set)
    _refuse_a_total_that_would_mean_two_things(tasks, learned_only)
    built = instantiate_offering(
        declared,
        learner_registry,
        model=model,
        tasks=tasks,
        losses=losses,
        learned_only=learned_only,
        teacher=teacher,
        loss=build_objective(declared),
        representation=build_representation(declared),
    )
    if not isinstance(built, Learner):
        raise TypeError(
            f"{declared.spelled!r} built {type(built).__name__}, which is not a Learner: it cannot turn a "
            "batch into a loss, and a trainer has nothing to ask it for."
        )
    return built


def teacher_heads(declared: TeacherConfig | None, heads: Mapping[str, HeadConfig]) -> Mapping[str, HeadConfig]:
    """The heads a teacher answers through: this run's own, and its own wherever it declared one.

    Its own, because the head is part of what makes a teacher a different network — a run continuing
    the tail of one needs the whole of that head here and the tail alone on the student, and a single
    declaration for both would leave the arrangement unwritable. The widths stay derived either way,
    which is what the rule against saying anything twice was about: the answer's width belongs to the
    task and the features' width to the teacher's own backbone.

    A head naming no stream reads the one this run's own head for that task reads, rather than the
    kind's default: the teacher answers the same task, so it reads the same kind of features, and
    resolving it from the run's own leaves that merge with the single home it already has.

    What a head inherited from this run does *not* bring with it is its ``checkpoint_path``. That file
    holds weights prepared for the student's head — most often the teacher's own head, carried over —
    and the teacher's weights arrive whole from its own ``checkpoint_path`` moments later. Inherited, it
    would be read and immediately written over, and it would refuse outright the day the two networks
    read features of different widths, which is the very arrangement this position exists for. A head
    the teacher declared for itself keeps whatever it was written with.
    """
    inherited = {name: _without_the_students_file(own) for name, own in heads.items()}
    if declared is None or declared.heads is None:
        return inherited
    unknown = sorted(set(declared.heads) - set(heads))
    if unknown:
        raise ValueError(
            f"`learner.teacher.heads` names {', '.join(unknown)}, and this run's tasks are "
            f"{', '.join(sorted(heads))}. A head for anything else would be built and then asked "
            f"nothing, all run, under a log that reads like any other's."
        )
    taught = dict(inherited)
    for name, own in declared.heads.items():
        taught[name] = own if own.stream is not None else own.model_copy(update={"stream": heads[name].stream})
    return taught


def _without_the_students_file(own: HeadConfig) -> HeadConfig:
    """One of this run's heads as a teacher inherits it: the same head, minus where the student's came from.

    The same declaration back where there was no file to drop, so a run that never wrote one goes on
    being handed the very heads it declared.
    """
    return own if own.checkpoint_path is None else own.model_copy(update={"checkpoint_path": None})


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
    if declared.import_path is not None and declared.heads is not None:
        raise ValueError(
            f"{declared.spelled!r} is a whole network and brings its own heads; drop `learner.teacher.heads`."
        )
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
