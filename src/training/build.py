"""Building what a run trains with: the learner over the assembled parts, and the two optimization factories."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from inspect import signature
from typing import TYPE_CHECKING, Any

from lightning.pytorch.profilers import Profiler
from torch.optim import Optimizer

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import LRSchedulerConfigType

from src.config import ComponentConfig, SchedulerConfig
from src.config.instantiate import fill_signature, instantiate, resolve_factory, resolve_params
from src.losses import Loss
from src.models import Model
from src.tasks import Task
from src.training.base import FitProfile, Learner, OptimizerFactory, SchedulerFactory
from src.training.registry import learner_registry, optimizer_registry, profiler_registry, scheduler_registry

LEARNING_RATE = "lr"
"""What a learning-rate graph is titled; Lightning's monitor reads it from the policy below.

Left unset, the monitor titles the graph after the optimizer's class — ``lr-AdamW/backbone`` — which
puts the reader's comparison under a name they are not comparing. Under ``lr`` the rates of every
group share one graph, each a line on it.
"""


def build_learner(
    declared: ComponentConfig, *, model: Model, tasks: Mapping[str, Task], losses: Mapping[str, Loss]
) -> Learner:
    """The algorithm a run trains by, over the parts it has already assembled."""
    built = instantiate(declared, learner_registry, model=model, tasks=tasks, losses=losses)
    if not isinstance(built, Learner):
        raise TypeError(
            f"{declared.spelled!r} built {type(built).__name__}, which is not a Learner: it cannot turn a "
            "batch into a loss, and a trainer has nothing to ask it for."
        )
    return built


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
