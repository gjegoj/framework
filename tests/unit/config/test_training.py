"""Loader and scheduler sections: forwarded knobs, and the ones we own."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import LoaderConfig, SchedulerConfig


def test_unknown_loader_keys_forward_to_torch() -> None:
    loader = LoaderConfig.model_validate({"num_workers": 4, "prefetch_factor": 2})

    assert loader.model_extra == {"prefetch_factor": 2}


def test_adapter_owned_loader_keys_are_rejected() -> None:
    """Per-stage shuffle is a convention, not a setting — configuring it would collide."""
    with pytest.raises(ValidationError, match="shuffle"):
        LoaderConfig.model_validate({"shuffle": True})


def test_drop_last_is_a_declared_loader_field() -> None:
    assert LoaderConfig().drop_last is False
    assert LoaderConfig.model_validate({"drop_last": True}).drop_last is True


def test_scheduler_policy_fields_stay_out_of_the_constructor_params() -> None:
    """Lightning's stepping policy is ours; everything else belongs to the scheduler."""
    scheduler = SchedulerConfig.model_validate({"name": "onecycle", "interval": "step", "max_lr": 0.1})

    assert scheduler.name == "onecycle"
    assert scheduler.interval == "step"
    assert scheduler.params == {"max_lr": 0.1}


def test_a_scheduler_may_also_be_named_by_import_path() -> None:
    scheduler = SchedulerConfig.model_validate({"_target_": "torch.optim.lr_scheduler.LinearLR"})

    assert scheduler.target == "torch.optim.lr_scheduler.LinearLR"
    assert scheduler.interval == "epoch"
