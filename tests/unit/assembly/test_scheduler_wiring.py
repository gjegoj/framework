"""Schedulers from config: the shipped groups are valid, and the two traps fail loudly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import OneCycleLR

from src.assembly.training import SCHEDULE_TITLE, build_scheduler_factory, fit_time_facts, per_group_rates
from src.config import ExperimentConfig, SchedulerConfig
from src.training import FitProfile
from tests.support.configs import paper_config

PROFILE = FitProfile(total_steps=5000, epochs=10)
BASE_LR = 1.0e-3
FAST_LR = 1.0e-2


def experiment(**scheduler: Any) -> ExperimentConfig:
    """The base experiment carrying the schedule under test."""
    return paper_config(scheduler=scheduler)


def optimizer() -> torch.optim.Optimizer:
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)


def grouped(*rates: float | None, base: float = BASE_LR) -> torch.optim.Optimizer:
    """One optimizer over named groups; ``None`` leaves a group on the base rate."""
    groups: list[dict[str, Any]] = []
    for index, rate in enumerate(rates):
        group: dict[str, Any] = {"params": [torch.nn.Parameter(torch.zeros(1))], "name": f"group{index}"}
        if rate is not None:
            group["lr"] = rate
        groups.append(group)
    return torch.optim.SGD(groups, lr=base)


def test_a_step_sized_schedule_on_the_epoch_clock_is_refused() -> None:
    """OneCycle sized for 5000 steps but stepped 10 times would hold the warm-up rate forever."""
    factory = build_scheduler_factory(experiment(name="onecycle", max_lr=0.01))
    assert factory is not None

    with pytest.raises(ValueError, match="interval: step"):
        factory(optimizer(), PROFILE)


def test_the_step_clock_sizes_the_schedule_from_the_fit() -> None:
    factory = build_scheduler_factory(experiment(name="onecycle", max_lr=0.01, interval="step"))
    assert factory is not None

    policy = factory(optimizer(), PROFILE)

    scheduler = policy["scheduler"]
    assert isinstance(scheduler, OneCycleLR)
    assert scheduler.total_steps == PROFILE.total_steps
    assert policy["interval"] == "step"


def test_a_plateau_without_a_monitor_is_refused_at_assembly() -> None:
    """Lightning would raise the same complaint at fit start, after data setup and model build."""
    with pytest.raises(ValueError, match="monitor"):
        build_scheduler_factory(experiment(name="plateau"))


def test_a_plateau_with_a_monitor_reaches_lightning_with_it() -> None:
    factory = build_scheduler_factory(experiment(name="plateau", monitor="val/loss", mode="min"))
    assert factory is not None

    policy = factory(optimizer(), PROFILE)

    assert policy["monitor"] == "val/loss"


def test_a_rate_a_task_declared_survives_the_schedule_that_sets_rates_outright() -> None:
    """The bug this exists for: a scalar `max_lr` broadcasts and the task's rate is gone.

    Measured before the fix — groups at 1e-3 and 1e-2 under `OneCycleLR(max_lr=1e-3)`
    both started at 4e-5 and both peaked at 1e-3 — and nothing in the run said so, so
    a head declared ten times faster trained at the backbone's pace for a whole fit.
    """
    factory = build_scheduler_factory(experiment(name="onecycle", max_lr=BASE_LR, interval="step"))
    assert factory is not None
    built = grouped(None, FAST_LR)

    factory(built, PROFILE)

    assert [group["initial_lr"] for group in built.param_groups] == [BASE_LR / 25, FAST_LR / 25]


def test_a_peak_above_the_base_rate_lifts_every_group_by_the_same_share() -> None:
    """A schedule declares a shape; which group runs how fast is the optimizer's answer.

    Handing each group its own rate verbatim would throw the shape away, so a peak
    declared at three times the base rate has to reach three times *every* group's.
    """
    spread = per_group_rates({"max_lr": 3 * BASE_LR}, grouped(None, FAST_LR))

    assert spread == {"max_lr": [3 * BASE_LR, 3 * FAST_LR]}


def test_a_band_keeps_its_shape_in_every_group() -> None:
    """Cyclic swings between two rates, and both ends belong to the group, not to one group."""
    spread = per_group_rates({"base_lr": BASE_LR, "max_lr": 4 * BASE_LR}, grouped(None, FAST_LR))

    assert spread == {"base_lr": [BASE_LR, FAST_LR], "max_lr": [4 * BASE_LR, 4 * FAST_LR]}


def test_a_rate_written_per_group_in_config_is_left_alone() -> None:
    """The user answered this question themselves, and a guess does not outrank an answer."""
    assert per_group_rates({"max_lr": [0.5, 0.25]}, grouped(None, FAST_LR)) == {}


def test_a_schedule_that_scales_the_rate_it_finds_is_never_touched() -> None:
    """Cosine and the rest read each group's own `lr`, so spreading one would be a fiction."""
    assert per_group_rates({"T_max": 10}, grouped(None, FAST_LR)) == {}


def test_a_single_group_needs_no_spreading() -> None:
    """Nothing to preserve, and a one-element list where config wrote a number is noise."""
    assert per_group_rates({"max_lr": BASE_LR}, optimizer()) == {}


def test_the_rates_reach_the_monitor_under_one_title() -> None:
    """Left unnamed, Lightning titles the graph after the optimizer's class and per group.

    The comparison a per-group rate is declared for — did the head move faster than
    the encoder — is exactly the one a title per group cannot make.
    """
    factory = build_scheduler_factory(experiment(name="onecycle", max_lr=BASE_LR, interval="step"))
    assert factory is not None

    policy = factory(grouped(None, FAST_LR), PROFILE)

    assert policy["name"] == SCHEDULE_TITLE


def shipped(name: str) -> dict[str, Any]:
    """One shipped group file, with the root anchors it interpolates against."""
    content = yaml.safe_load(Path(f"configs/scheduler/{name}.yaml").read_text(encoding="utf-8"))
    resolved = OmegaConf.to_container(
        OmegaConf.create({"lr": 1.0e-3, "epochs": 10, "scheduler": content}), resolve=True
    )
    assert isinstance(resolved, dict)
    section = resolved["scheduler"]
    assert isinstance(section, dict)
    return section


@pytest.mark.parametrize("name", ["cosine", "onecycle", "plateau", "step"])
def test_every_shipped_group_validates_and_builds(name: str) -> None:
    """The files a user swaps in must hold real declarations, not examples that drift."""
    declared = shipped(name)

    factory = build_scheduler_factory(experiment(**declared))
    assert factory is not None

    policy = factory(optimizer(), PROFILE)
    assert policy["scheduler"] is not None


def test_the_shipped_onecycle_group_declares_the_step_clock() -> None:
    """The exact trap the guard exists for must not ship in our own file."""
    assert SchedulerConfig.model_validate(shipped("onecycle")).interval == "step"


def test_the_none_group_clears_the_section() -> None:
    content = yaml.safe_load(Path("configs/scheduler/none.yaml").read_text(encoding="utf-8"))

    assert content == {"scheduler": None}


class EpochScheduler:
    """A scheduler that counts epochs instead of steps (no ``total_steps``)."""

    def __init__(self, optimizer: Any, epochs: int = 1, steps_per_epoch: int = 1) -> None:
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch


def test_the_most_precise_fact_is_the_only_one_filled() -> None:
    """A scheduler declaring ``total_steps`` needs nothing else."""
    assert fit_time_facts(OneCycleLR, PROFILE, configured={"max_lr": 0.1}) == {"total_steps": 5000}


def test_epoch_shaped_schedulers_get_the_epoch_facts() -> None:
    assert fit_time_facts(EpochScheduler, PROFILE, configured={}) == {"steps_per_epoch": 500, "epochs": 10}


def test_a_configured_value_is_never_overwritten() -> None:
    """Environment facts fill what config left unset; they do not contradict it."""
    assert fit_time_facts(OneCycleLR, PROFILE, configured={"max_lr": 0.1, "total_steps": 42}) == {}


def test_schedulers_without_fit_time_params_get_nothing() -> None:
    assert fit_time_facts(torch.optim.lr_scheduler.CosineAnnealingLR, PROFILE, configured={"T_max": 10}) == {}
