"""What a run optimizes with: the optimizer over named groups, and the schedule sized from the fit itself."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn
from torch.optim import Optimizer

from src.config import ComponentConfig, SchedulerConfig
from src.core import Batch, StepOutput, TargetInfo
from src.losses.build import build_loss
from src.tasks import Classification, MetricLearning
from src.training import StandardLearner
from src.training.base import FitProfile, Learner, ParameterGroup
from src.training.build import build_learner, build_optimizer_factory, build_scheduler_factory
from src.training.registry import optimizer_registry, scheduler_registry
from tests.support.models import Echo

RATE = 1e-3
PROFILE = FitProfile(total_steps=100, epochs=5)
IDENTITIES = {0: "ann", 1: "bob"}


def identities() -> Batch:
    """A batch of two, each some identity the training split showed."""
    return Batch(inputs={}, targets={"identity": torch.tensor([0, 1])}, count=2)


def groups(**rates: float) -> list[ParameterGroup]:
    """One group per name; a name with a rate of its own carries it, as a task's declared lr does."""
    built: list[ParameterGroup] = []
    for name, rate in rates.items():
        group: ParameterGroup = {"name": name, "params": [nn.Parameter(torch.zeros(2))]}
        if rate > 0:
            group["lr"] = rate
        built.append(group)
    return built


def scheduled(declared: dict[str, Any], optimizer: Optimizer, profile: FitProfile = PROFILE) -> dict[str, Any]:
    factory = build_scheduler_factory(SchedulerConfig.model_validate(declared))
    assert factory is not None
    return dict(factory(optimizer, profile))


class TestOptimizer:
    def test_the_runs_rate_is_the_base_and_a_group_keeps_the_one_it_declared(self) -> None:
        """The base rate has one home, `experiment.lr`; a task's own rate rides on its group."""
        optimizer = build_optimizer_factory(ComponentConfig(name="adamw"), RATE)(groups(backbone=0, head=1e-2))

        assert [group["lr"] for group in optimizer.param_groups] == [RATE, 1e-2]

    def test_declared_options_reach_the_optimizer(self) -> None:
        optimizer = build_optimizer_factory(ComponentConfig.model_validate({"name": "sgd", "momentum": 0.9}), RATE)(
            groups(backbone=0)
        )

        assert optimizer.param_groups[0]["momentum"] == 0.9

    def test_a_tasks_declared_rate_reaches_its_own_parameters_and_nothing_else(self) -> None:
        """The whole point of the groups: one rate for the encoder, another for what the task owns."""
        task = Classification("species", TargetInfo(classes={0: "cat", 1: "dog"}), lr=1e-2)
        trained = StandardLearner(
            Echo({"species": torch.zeros(2, 2)}),
            {"species": task},
            {"species": build_loss(task.default_loss, task.facts())},
        )

        optimizer = build_optimizer_factory(ComponentConfig(name="adamw"), RATE)(trained.parameter_groups())

        assert {group["name"]: group["lr"] for group in optimizer.param_groups} == {"backbone": RATE, "species": 1e-2}

    def test_an_optimizer_torch_offers_needs_no_registration(self) -> None:
        factory = build_optimizer_factory(ComponentConfig.model_validate({"_target_": "torch.optim.RMSprop"}), RATE)

        assert isinstance(factory(groups(backbone=0)), torch.optim.RMSprop)


SCHEDULES: dict[str, dict[str, Any]] = {
    "cosine": {"T_max": 7},
    "onecycle": {"interval": "step"},
    "plateau": {"monitor": "val/loss"},
    "step": {"step_size": 3},
}
"""The least each registered schedule needs written for it: a newly registered name needs a row here."""


class TestRegistries:
    @pytest.mark.parametrize("name", list(optimizer_registry))
    def test_every_registered_optimizer_moves_the_groups_it_is_given(self, name: str) -> None:
        built = build_optimizer_factory(ComponentConfig(name=name), RATE)(groups(backbone=0, head=1e-2))

        assert [group["lr"] for group in built.param_groups] == [RATE, 1e-2]

    @pytest.mark.parametrize("name", list(scheduler_registry))
    def test_every_registered_schedule_is_built_from_the_fit_it_will_run(self, name: str) -> None:
        assert name in SCHEDULES, f"{name!r} is registered but has no declaration; add a row to SCHEDULES."
        optimizer = build_optimizer_factory(ComponentConfig(name="adamw"), RATE)(groups(backbone=0))

        policy = scheduled({"name": name, **SCHEDULES[name]}, optimizer)

        assert policy["scheduler"] is not None and policy["name"] == "lr"


class TestSchedule:
    @pytest.fixture
    def optimizer(self) -> Optimizer:
        return build_optimizer_factory(ComponentConfig(name="adamw"), RATE)(groups(backbone=0, head=1e-2))

    def test_nothing_declared_is_no_schedule(self) -> None:
        assert build_scheduler_factory(None) is None

    def test_a_step_clocked_schedule_is_sized_from_the_fit_it_will_run(self, optimizer: Optimizer) -> None:
        """`total_steps` is the fit's own number: accumulation, drop_last and limits are already in it."""
        policy = scheduled({"name": "onecycle", "interval": "step"}, optimizer)

        assert policy["scheduler"].total_steps == PROFILE.total_steps
        assert (policy["interval"], policy["frequency"]) == ("step", 1)

    def test_the_peak_follows_each_group_so_a_task_keeps_its_pace(self, optimizer: Optimizer) -> None:
        """A single peak would flatten every group onto one rate and quietly undo a task's declared lr."""
        scheduled({"name": "onecycle", "interval": "step"}, optimizer)

        assert [group["max_lr"] for group in optimizer.param_groups] == [RATE, 1e-2]

    def test_a_peak_written_by_hand_is_taken_as_written(self, optimizer: Optimizer) -> None:
        scheduled({"name": "onecycle", "interval": "step", "max_lr": 0.5}, optimizer)

        assert [group["max_lr"] for group in optimizer.param_groups] == [0.5, 0.5]

    def test_an_epoch_clocked_schedule_is_left_to_its_own_declaration(self, optimizer: Optimizer) -> None:
        """`T_max` is cosine's own name for its length, not a fact every schedule shares; a run states it."""
        policy = scheduled({"name": "cosine", "T_max": 7}, optimizer)

        assert policy["scheduler"].T_max == 7 and policy["interval"] == "epoch"

    def test_a_schedule_sized_in_steps_but_stepped_by_epoch_is_refused(self, optimizer: Optimizer) -> None:
        """It would advance 5 of 100 steps and hold the warm-up rate for the whole run."""
        with pytest.raises(ValueError, match="interval"):
            scheduled({"name": "onecycle"}, optimizer)

    def test_a_plateau_with_no_metric_to_watch_is_refused(self) -> None:
        with pytest.raises(ValueError, match="monitor"):
            build_scheduler_factory(SchedulerConfig.model_validate({"name": "plateau"}))

    def test_a_plateau_watches_the_metric_it_was_given(self, optimizer: Optimizer) -> None:
        policy = scheduled({"name": "plateau", "monitor": "val/loss", "mode": "min"}, optimizer)

        assert policy["monitor"] == "val/loss" and policy["scheduler"].mode == "min"


class TestLearner:
    def test_the_name_a_run_declares_is_built_over_the_parts_it_assembled(self) -> None:
        task = Classification("species", TargetInfo(classes={0: "cat", 1: "dog"}))

        built = build_learner(
            ComponentConfig(name="standard"),
            model=Echo({"species": torch.zeros(2, 2)}),
            tasks={"species": task},
            losses={"species": build_loss(task.default_loss, task.facts())},
        )

        assert isinstance(built, StandardLearner) and set(built.tasks) == {"species"}

    def test_an_algorithm_that_holds_no_task_losses_is_built_without_them(self) -> None:
        """`Learner` asks for a model and tasks; an algorithm that owns its objective differently —
        a distilled one, a self-supervised one — must be buildable against the contract it implements
        rather than against the one the standard learner happens to have."""
        declared = ComponentConfig.model_validate({"_target_": "tests.unit.training.test_build.Lonely"})
        task = Classification("species", TargetInfo(classes={0: "cat", 1: "dog"}))
        losses = {"species": build_loss(task.default_loss, task.facts())}

        built = build_learner(declared, model=Echo({}), tasks={"species": task}, losses=losses)

        assert isinstance(built, Lonely) and set(built.tasks) == {"species"} and built.loss_of("species") is None

    def test_a_task_judged_on_identities_it_never_learned_is_scored_while_learning_and_not_after(self) -> None:
        """Derived from what the data settled, not declared: the encoder said its vocabulary is open."""
        task = MetricLearning("identity", TargetInfo(classes=IDENTITIES, open_set=True), embedding_dim=4)

        built = build_learner(
            ComponentConfig(name="standard"),
            model=Echo({"identity": torch.zeros(2, 4)}),
            tasks={"identity": task},
            losses={"identity": build_loss(task.default_loss, task.facts())},
        )

        assert built.eval().step(identities()).loss is None
        assert built.train().step(identities()).loss is not None

    def test_a_run_whose_total_would_mean_two_things_in_two_stages_is_refused_by_name(self) -> None:
        """One objective stops in evaluation and one does not, so `val/loss` would total less than
        `train/loss` under the same name, on the same chart, with nothing saying so."""
        open_set = MetricLearning("identity", TargetInfo(classes=IDENTITIES, open_set=True), embedding_dim=4)
        closed = Classification("species", TargetInfo(classes={0: "cat", 1: "dog"}))
        tasks = {"identity": open_set, "species": closed}

        with pytest.raises(ValueError, match="identity"):
            build_learner(
                ComponentConfig(name="standard"),
                model=Echo({"identity": torch.zeros(2, 4), "species": torch.zeros(2, 2)}),
                tasks=tasks,
                losses={name: build_loss(one.default_loss, one.facts()) for name, one in tasks.items()},
            )

    def test_something_that_cannot_take_a_step_is_refused_where_it_was_declared(self) -> None:
        declared = ComponentConfig.model_validate({"_target_": "tests.unit.training.test_build.Bare"})

        with pytest.raises(TypeError, match="Learner"):
            build_learner(declared, model=Echo({}), tasks={}, losses={})


class Lonely(Learner):
    """A learner built on nothing but the contract: a model, its tasks, and a step."""

    def step(self, batch: Batch) -> StepOutput:
        raise NotImplementedError


class Bare:
    """Something a `_target_` may reach that cannot turn a batch into a loss."""

    def __init__(self, **parts: Any) -> None:
        self.parts = parts
