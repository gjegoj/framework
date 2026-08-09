"""Optimizer and scheduler registries: torch classes under config-facing names."""

from __future__ import annotations

import torch

from src.training.registry import optimizer_registry, scheduler_registry


def test_built_in_optimizers_are_registered() -> None:
    assert set(optimizer_registry) == {"adamw", "adam", "sgd"}


def test_built_in_schedulers_are_registered() -> None:
    assert set(scheduler_registry) == {"cosine", "onecycle", "plateau", "step"}


def test_create_builds_a_configured_optimizer() -> None:
    parameter = torch.nn.Parameter(torch.zeros(2))

    built = optimizer_registry.create("adamw", params=[parameter], lr=0.5)

    assert isinstance(built, torch.optim.AdamW)
    assert built.param_groups[0]["lr"] == 0.5


def test_create_builds_a_configured_scheduler() -> None:
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(2))], lr=0.1)

    built = scheduler_registry.create("step", optimizer=optimizer, step_size=5)

    assert isinstance(built, torch.optim.lr_scheduler.StepLR)
