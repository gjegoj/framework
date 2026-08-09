"""Per-task learning rates: a task's own bricks may learn at their own pace."""

from __future__ import annotations

import copy
from functools import partial

import pytest
import torch

from src.assembly.tasks import build_tasks
from src.config import load_config
from src.core import DataProfile, Objective, TargetFacts, Task, Topology
from src.losses import CrossEntropyCriterion, ProxyAngularCriterion
from src.models import CompositeModel, IdentityHead, LinearHead, TaskComponents
from src.tasks.activations import identity, softmax_probabilities
from src.tasks.adapters import as_class_indices
from src.training import OptimizerFactory, TrainingModule
from src.training.module import SHARED_GROUP
from tests.support.fakes import FlattenBackbone, LearningBackbone

BASE_LR = 1.0e-3
FAST_LR = 1.0e-2
FEATURES = 12


def task(name: str, lr: float | None = None) -> Task:
    return Task(name=name, topology=Topology.GLOBAL, objective=Objective.MULTICLASS, metrics={}, lr=lr)


def module(tasks: list[Task], factory: OptimizerFactory | None = None) -> TrainingModule:
    components = {
        one.name: TaskComponents(
            head=LinearHead(FEATURES, 3),
            criterion=CrossEntropyCriterion(),
            activation=softmax_probabilities,
            target_adapter=as_class_indices,
        )
        for one in tasks
    }
    return TrainingModule(
        model=CompositeModel(backbone=LearningBackbone(dim=FEATURES), components=components),
        tasks=tasks,
        optimizer_factory=factory or partial(torch.optim.SGD, lr=BASE_LR),
    )


def test_a_task_with_its_own_lr_gets_its_own_named_group() -> None:
    built = module([task("label", lr=FAST_LR), task("kind")])

    optimizer = built.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)

    named = {group.get("name"): group for group in optimizer.param_groups}
    assert named["label"]["lr"] == FAST_LR


def test_the_shared_group_follows_the_optimizer_default() -> None:
    """The base rate has one home — the optimizer section — and the shared group inherits it."""
    built = module([task("label", lr=FAST_LR), task("kind")])

    optimizer = built.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)

    shared = [group for group in optimizer.param_groups if group["name"] == SHARED_GROUP]
    assert len(shared) == 1
    assert shared[0]["lr"] == BASE_LR


def test_criterion_parameters_learn_at_their_tasks_rate() -> None:
    """A proxy's prototypes are the task's own state, like its head — not backbone."""
    metric_task = Task(name="person", topology=Topology.GLOBAL, objective=Objective.METRIC, metrics={}, lr=FAST_LR)
    proxy = ProxyAngularCriterion(num_classes=3, embedding_dim=FEATURES)
    built = TrainingModule(
        model=CompositeModel(
            backbone=FlattenBackbone(dim=FEATURES),
            components={
                "person": TaskComponents(
                    head=IdentityHead(),
                    criterion=proxy,
                    activation=identity,
                    target_adapter=as_class_indices,
                )
            },
        ),
        tasks=[metric_task],
        optimizer_factory=partial(torch.optim.SGD, lr=BASE_LR),
    )

    optimizer = built.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)

    named = {group.get("name"): group for group in optimizer.param_groups}
    fast_ids = {id(parameter) for parameter in named["person"]["params"]}
    assert id(proxy.prototypes) in fast_ids


def test_every_group_is_named_even_where_no_rate_was_declared() -> None:
    """The groups are what a learning-rate monitor draws, so a run shows every pace.

    Named only where a rate was overridden, a run declaring none draws one anonymous
    line and a reader cannot see that the head and the encoder move together.
    """
    built = module([task("label"), task("kind")])

    optimizer = built.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)

    assert [group["name"] for group in optimizer.param_groups] == [SHARED_GROUP, "label", "kind"]
    assert all(group["lr"] == BASE_LR for group in optimizer.param_groups)


def test_splitting_the_groups_moves_the_weights_exactly_as_one_group_would() -> None:
    """The split is drawn for a chart, so it has to cost the run nothing.

    A group that quietly carried a rate or a decay of its own would train a different
    model for the sake of a drawing, which is the one way this change could do harm.
    Driven by identical gradients rather than a forward pass, so the two runs differ
    in nothing but how their parameters are grouped.
    """
    grouped = module([task("label"), task("kind")], partial(torch.optim.AdamW, lr=BASE_LR, weight_decay=1.0e-2))
    flat = copy.deepcopy(grouped)
    split = grouped.configure_optimizers()
    assert isinstance(split, torch.optim.Optimizer)
    whole = torch.optim.AdamW(flat.parameters(), lr=BASE_LR, weight_decay=1.0e-2)
    gradients = [torch.randn_like(parameter) for parameter in grouped.parameters()]

    for _ in range(5):
        for built, optimizer in ((grouped, split), (flat, whole)):
            for parameter, gradient in zip(built.parameters(), gradients, strict=True):
                parameter.grad = gradient.clone()
            optimizer.step()

    moved = zip(grouped.parameters(), flat.parameters(), strict=True)
    assert all(torch.equal(one, other) for one, other in moved)


def test_every_parameter_is_assigned_exactly_once() -> None:
    """torch refuses duplicates; a lost parameter would silently never train."""
    built = module([task("label", lr=FAST_LR), task("kind")])

    optimizer = built.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)

    assigned = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(assigned) == len(set(assigned))
    assert len(assigned) == sum(1 for _ in built.model.parameters())


def test_a_rate_for_a_task_without_own_parameters_is_refused() -> None:
    """A silently ignored lr is worse than none: the run would train at the wrong pace."""
    built = module([task("label")])
    built._tasks.append(task("ghost", lr=FAST_LR))

    with pytest.raises(ValueError, match="ghost"):
        built.configure_optimizers()


def test_the_declared_rate_travels_from_yaml_to_the_task() -> None:
    config = load_config(
        {
            "data": {
                "source": "a.csv",
                "split": {"train": 0.6, "val": 0.2, "test": 0.2},
                "inputs": {"image": {"column": "image"}},
            },
            "tasks": {"label": {"preset": "classification", "target": "label", "lr": FAST_LR}},
            "model": {"name": "timm", "model_name": "resnet18"},
        }
    )
    profile = DataProfile()
    profile.record("label", TargetFacts(num_classes=3))

    tasks, _ = build_tasks(config, profile, FlattenBackbone(dim=FEATURES))

    assert tasks[0].lr == FAST_LR


def test_a_non_positive_rate_is_refused_by_the_entity() -> None:
    with pytest.raises(ValueError, match="lr"):
        task("label", lr=0.0)
