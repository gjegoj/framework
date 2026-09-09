"""Building what a run optimizes with: the factories over the optimizer and scheduler sections."""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, Any, Final

from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.config.instantiate import fill_signature, resolve_target
from src.training.optim import FitProfile, OptimizerFactory, SchedulerFactory
from src.training.registry import optimizer_registry, scheduler_registry

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lightning.pytorch.utilities.types import LRSchedulerConfigType
    from torch.optim import Optimizer

    from src.config import OptimizerConfig, SchedulerConfig

log = logging.getLogger(__name__)

SCHEDULE_TITLE: Final = "lr"
"""What a learning-rate graph is called, given to Lightning's monitor as the schedule's name.

Left unset the monitor titles the graph after the optimizer's class and writes
``lr-AdamW/backbone``, which puts a stage-less family under a title naming
something a reader is not comparing. Under ``lr`` the key becomes ``lr/backbone``
— the grammar's own shape, so the rates share one graph and each group is a
series on it.
"""

GROUP_RATES: Final = ("max_lr", "base_lr")
"""Scheduler parameters naming an *absolute* learning rate per parameter group.

The rule is the parameter's name, not the scheduler's registry key — the same
reading ``fit_time_facts`` gives ``total_steps``. A key is the wrong thing to
match on twice over: the grammar lets a schedule arrive by ``_target_`` and carry
no key at all, and a table of keys has to be extended by hand for every schedule
added, failing silently when it is not.
"""


def build_optimizer_factory(declared: OptimizerConfig) -> OptimizerFactory:
    """A factory, not an instance: the model's parameters do not exist yet."""
    return partial(resolve_target(declared, optimizer_registry), **declared.params)


def fit_time_facts(
    scheduler_class: Callable[..., Any],
    profile: FitProfile,
    configured: Mapping[str, Any],
) -> dict[str, int]:
    """The fit-time facts a scheduler declares and the config left unset.

    Only canonical torch names are filled (``total_steps`` means one thing), so no
    per-scheduler mapping exists; measured, ``OneCycleLR`` accepts all three and lets
    ``total_steps`` win. Returned rather than applied, because the scheduler needs the
    optimizer first and the caller reads which facts were filled. Config wins over these
    facts: the length of a schedule is the environment's estimate, and shaping it
    deliberately is a real recipe. Filled by ``fill_signature``, the exception for
    constructors that are not ours (ADR-0004).

    Parameters:
        scheduler_class (Callable): The scheduler constructor about to be called.
        profile (FitProfile): Facts read from the trainer.
        configured (Mapping[str, Any]): Parameters already supplied by config.
    """
    precise = fill_signature(scheduler_class, total_steps=profile.total_steps)
    facts = precise or fill_signature(scheduler_class, steps_per_epoch=profile.steps_per_epoch, epochs=profile.epochs)
    return {name: value for name, value in facts.items() if name not in configured}


def per_group_rates(configured: Mapping[str, Any], optimizer: Optimizer) -> dict[str, list[float]]:
    """A declared rate spread over the optimizer's groups, each keeping its own pace.

    ``OneCycleLR`` and ``CyclicLR`` set a rate outright, and a scalar broadcasts to every
    group — measured, ``OneCycleLR(max_lr=3e-4)`` over groups at 3e-4 and 1e-2 peaks both
    at 3e-4, and the task's declared rate is gone. Each group gets the declared value scaled
    by how its rate compares with the optimizer's own. A list already written in config is
    left alone.
    """
    groups = optimizer.param_groups
    if len(groups) <= 1:
        return {}
    base = optimizer.defaults["lr"]
    ratios = [group["lr"] / base for group in groups]
    return {
        name: [configured[name] * ratio for ratio in ratios]
        for name in GROUP_RATES
        if isinstance(configured.get(name), (int, float))
    }


def _spread_as_written(spread: Mapping[str, list[float]], optimizer: Optimizer) -> str:
    """The spread rates as the kwarg they become, each value under the group's name.

    Said out loud because config declares one number and the run trains on several.
    """
    names = [group.get("name", f"group {index}") for index, group in enumerate(optimizer.param_groups, start=1)]
    return ", ".join(
        f"{parameter}=[" + ", ".join(f"{name} {value:.2e}" for name, value in zip(names, values, strict=True)) + "]"
        for parameter, values in spread.items()
    )


def build_scheduler_factory(declared: SchedulerConfig | None) -> SchedulerFactory | None:
    """A factory too: a scheduler needs the optimizer and the fit-time facts.

    Its own knobs come from config; the canonical fit-time facts
    (``total_steps`` and friends) are filled only where the class declares them
    and config left them unset, so no per-scheduler mapping has to be kept.
    """
    if declared is None:
        return None
    scheduler_class = resolve_target(declared, scheduler_registry)
    if (
        isinstance(scheduler_class, type)
        and issubclass(scheduler_class, ReduceLROnPlateau)
        and declared.monitor is None
    ):
        raise ValueError(
            "A plateau schedule reacts to a logged metric, so it needs 'monitor' — "
            "e.g. scheduler: {name: plateau, monitor: val/loss, mode: min}."
        )

    def factory(optimizer: Optimizer, profile: FitProfile) -> LRSchedulerConfigType:
        params = declared.params
        derived = fit_time_facts(scheduler_class, profile, params)
        step_clocked = {"total_steps", "steps_per_epoch"} & derived.keys()
        if step_clocked and declared.interval == "epoch":
            filled = ", ".join(sorted(step_clocked))
            raise ValueError(
                f"{scheduler_class.__name__} was sized in optimizer steps ({filled} filled from the "
                f"fit), but interval is 'epoch' — it would advance {profile.epochs} of "
                f"{profile.total_steps} steps and hold the warm-up rate for the whole run. "
                f"Set interval: step."
            )
        spread = per_group_rates(params, optimizer)
        if spread:
            log.info("%s takes its rate per group: %s", scheduler_class.__name__, _spread_as_written(spread, optimizer))
        scheduler = scheduler_class(optimizer, **{**params, **spread}, **derived)
        policy: dict[str, Any] = {
            "scheduler": scheduler,
            "name": SCHEDULE_TITLE,
            "interval": declared.interval,
            "frequency": declared.frequency,
            "strict": declared.strict,
        }
        if declared.monitor is not None:
            policy["monitor"] = declared.monitor
        return policy  # type: ignore[return-value]

    return factory
