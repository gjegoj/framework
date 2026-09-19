"""What a run optimizes with: the optimizer over named groups, and the schedule sized from the fit itself."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, override

import pytest
import torch
from torch import nn
from torch.optim import Optimizer

from src.config import ComponentConfig, LearnerConfig, SchedulerConfig
from src.core import Batch, Representation, StepOutput, TargetInfo
from src.losses import KullbackLeibler
from src.losses.build import build_loss
from src.models import Model
from src.tasks import Classification, MetricLearning
from src.training import StandardLearner
from src.training.base import FitProfile, Learner, ParameterGroup
from src.training.build import (
    build_learner,
    build_optimizer_factory,
    build_scheduler_factory,
    refuse_a_learner_and_its_child_positions_that_disagree,
)
from src.training.distillation import DistillationLearner
from src.training.registry import optimizer_registry, scheduler_registry
from tests.support.models import Angles, Echo

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


SOFT: dict[str, Any] = {"name": "kullback_leibler", "temperature": 2.0, "scale": 8.0}
"""An objective against a teacher, declared with both its knobs so either being lost is visible.

Carrying a scale, it reads the angles a `cosine` head answers with — which is what `ANGULAR` answers.
"""

PLAIN: dict[str, Any] = {"name": "kullback_leibler"}
"""The same objective over the projections an ordinary head answers with: no scale, nothing to convert."""

ANGULAR = Angles("species", "image", 4, 2)
"""A network answering in angles, for the half of the pair a plain objective cannot read."""


class Mixed(Echo):
    """A run whose heads answer in two spaces, as one pairing a `cosine` head with an ordinary one does."""

    @override
    def produces(self, task: str) -> Representation:
        return Representation.COSINES if task == "identity" else Representation.PROJECTED


def distilled(declared: LearnerConfig, answering: Model | None = None, over: Sequence[str] = ("species",)) -> Learner:
    """One task over one network with a second beside it, assembled the way a run assembles them."""
    tasks = {name: Classification(name, TargetInfo(classes={0: "cat", 1: "dog"})) for name in over}
    answers = {name: torch.zeros(2, 2) for name in over}
    return build_learner(
        declared,
        model=answering if answering is not None else Echo(answers),
        tasks=tasks,
        losses={name: build_loss(task.default_loss, task.facts()) for name, task in tasks.items()},
        teacher=Echo(answers),
    )


class TestLearner:
    def test_the_name_a_run_declares_is_built_over_the_parts_it_assembled(self) -> None:
        task = Classification("species", TargetInfo(classes={0: "cat", 1: "dog"}))

        built = build_learner(
            LearnerConfig(name="standard"),
            model=Echo({"species": torch.zeros(2, 2)}),
            tasks={"species": task},
            losses={"species": build_loss(task.default_loss, task.facts())},
        )

        assert isinstance(built, StandardLearner) and set(built.tasks) == {"species"}

    def test_an_algorithm_that_holds_no_task_losses_is_built_without_them(self) -> None:
        """`Learner` asks for a model and tasks; an algorithm that owns its objective differently —
        a distilled one, a self-supervised one — must be buildable against the contract it implements
        rather than against the one the standard learner happens to have."""
        declared = LearnerConfig.model_validate({"_target_": "tests.unit.training.test_build.Lonely"})
        task = Classification("species", TargetInfo(classes={0: "cat", 1: "dog"}))
        losses = {"species": build_loss(task.default_loss, task.facts())}

        built = build_learner(declared, model=Echo({}), tasks={"species": task}, losses=losses)

        assert isinstance(built, Lonely) and set(built.tasks) == {"species"} and built.loss_of("species") is None

    def test_a_task_judged_on_identities_it_never_learned_is_scored_while_learning_and_not_after(self) -> None:
        """Derived from what the data settled, not declared: the encoder said its vocabulary is open."""
        task = MetricLearning("identity", TargetInfo(classes=IDENTITIES, open_set=True), embedding_dim=4)

        built = build_learner(
            LearnerConfig(name="standard"),
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
                LearnerConfig(name="standard"),
                model=Echo({"identity": torch.zeros(2, 4), "species": torch.zeros(2, 2)}),
                tasks=tasks,
                losses={name: build_loss(one.default_loss, one.facts()) for name, one in tasks.items()},
            )

    def test_something_that_cannot_take_a_step_is_refused_where_it_was_declared(self) -> None:
        declared = LearnerConfig.model_validate({"_target_": "tests.unit.training.test_build.Bare"})

        with pytest.raises(TypeError, match="Learner"):
            build_learner(declared, model=Echo({}), tasks={}, losses={})

    def test_the_objective_a_learner_adds_beside_the_tasks_is_declared_in_a_position_of_its_own(self) -> None:
        """A child position, resolved against the registry that serves it, as `model.backbone` is.

        Written below the learner rather than as arguments of it, because what varies is a whole
        component — an objective with its own knobs and its own statement of what it reads — and the
        one grammar this framework has for that is a position with a registry behind it.
        """
        declared = LearnerConfig.model_validate({"name": "distillation", "loss": SOFT})

        built = distilled(declared, answering=Angles("species", "image", 4, 2))

        assert isinstance(built, DistillationLearner)
        assert isinstance(built.loss, KullbackLeibler)
        assert (built.loss.temperature, built.loss.scale) == (2.0, 8.0)

    def test_an_objective_named_from_the_registry_that_serves_another_position_is_refused(self) -> None:
        """A registry belongs to a position: what a task is judged by and what a teacher is agreed with
        are different questions, and a name answering one of them answers nothing about the other."""
        declared = LearnerConfig.model_validate({"name": "distillation", "loss": {"name": "focal"}})

        with pytest.raises(LookupError, match="Unknown distillation loss 'focal'"):
            distilled(declared)

    def test_an_objective_declared_on_a_learner_that_reads_none_is_refused_rather_than_dropped(self) -> None:
        """Offered facts a constructor does not name are dropped by contract; a *declaration* is not one.

        Asked of the constructor rather than of the name, for the reason a teacher is: a `_target_`
        declaration writes no name, so keying on one would let the very learners that most need saying
        so pass in silence.
        """
        declared = LearnerConfig.model_validate({"name": "standard", "loss": SOFT})

        with pytest.raises(ValueError, match=r"learner\.loss.*distillation"):
            refuse_a_learner_and_its_child_positions_that_disagree(declared)

    @pytest.mark.parametrize(
        ("answering", "declared"),
        [
            pytest.param(None, SOFT, id="an objective reading angles over a head that projects"),
            pytest.param(ANGULAR, PLAIN, id="an objective reading projections over a head that answers in angles"),
            pytest.param(ANGULAR, None, id="the objective a run gets without declaring one, over angles"),
        ],
    )
    def test_an_objective_that_reads_what_the_heads_never_answer_with_is_refused_before_a_step(
        self, answering: Model | None, declared: dict[str, Any] | None
    ) -> None:
        """Nothing in a tensor tells a projection from an angle, so this pair would train and report a number.

        Measured on eight real classes: a divergence over unscaled cosines is 256 times the one over the
        same two answers read as the objective beside them reads them — a term that looks like work and
        descends almost nothing, under a log like any other run's.

        The undeclared row is the one that matters most: every distilling config written before this
        position existed writes no objective at all, so a check that ran only for the written form would
        pass exactly the runs most likely to be wrong.
        """
        with pytest.raises(ValueError, match=r"learner\.loss"):
            distilled(LearnerConfig.model_validate({"name": "distillation", "loss": declared}), answering=answering)

    def test_one_objective_over_heads_that_answer_differently_is_refused_naming_the_task(self) -> None:
        """A run may pair a `cosine` head with an ordinary one; a single term cannot read both.

        The task is named rather than the run, because that is the line a reader has to change.
        """
        declared = LearnerConfig.model_validate({"name": "distillation", "loss": PLAIN})
        answers = {name: torch.zeros(2, 2) for name in ("species", "identity")}

        with pytest.raises(ValueError, match=r"'identity'.*learner\.loss"):
            distilled(declared, answering=Mixed(answers), over=("species", "identity"))

    def test_something_that_compares_no_two_answers_is_refused_where_it_was_declared(self) -> None:
        """A `_target_` reaches anything at all; what this position takes reads two answers and reports one."""
        declared = LearnerConfig.model_validate({"name": "distillation", "loss": {"_target_": "torch.nn.Identity"}})

        with pytest.raises(TypeError, match="does not compare"):
            distilled(declared)

    def test_a_knob_that_moved_onto_the_objective_is_refused_where_it_used_to_be_written(self) -> None:
        """`temperature` was the learner's; it belongs to the term that softens, and only one may hold it."""
        declared = LearnerConfig.model_validate({"name": "distillation", "temperature": 4.0})

        with pytest.raises(ValueError, match="takes no temperature"):
            distilled(declared)


class Lonely(Learner):
    """A learner built on nothing but the contract: a model, its tasks, and a step."""

    def step(self, batch: Batch) -> StepOutput:
        raise NotImplementedError


class Bare:
    """Something a `_target_` may reach that cannot turn a batch into a loss."""

    def __init__(self, **parts: Any) -> None:
        self.parts = parts
