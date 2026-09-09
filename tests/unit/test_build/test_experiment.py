"""``build``: config in, a ready experiment out — in the one order allowed."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import lightning as L
import torch

from src.build import build
from src.tasks.kinds import Classification
from src.training import FitProfile, TrainingData, TrainingModule
from src.training.build import build_optimizer_factory, build_scheduler_factory
from src.training.module import SHARED_GROUP
from tests.support.configs import TASKS, disk_config
from tests.support.narrowing import tensor


def test_the_optimizer_factory_carries_configured_params(dataset_root: Path) -> None:
    config = disk_config(dataset_root, optimizer={"name": "sgd", "lr": 0.5})
    parameter = torch.nn.Parameter(torch.zeros(2))

    optimizer = build_optimizer_factory(config.optimizer)([{"params": [parameter], "name": SHARED_GROUP}])

    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["lr"] == 0.5


def test_no_scheduler_section_means_no_scheduler(dataset_root: Path) -> None:
    assert build_scheduler_factory(disk_config(dataset_root).scheduler) is None


def test_a_scheduler_gets_its_fit_time_facts_filled(dataset_root: Path) -> None:
    """The user writes no `total_steps`: the environment supplies what it knows."""
    config = disk_config(dataset_root, scheduler={"name": "onecycle", "interval": "step", "max_lr": 0.1})
    factory = build_scheduler_factory(config.scheduler)
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(2))], lr=0.1)

    assert factory is not None
    policy = factory(optimizer, FitProfile(total_steps=50, epochs=10))

    assert isinstance(policy["scheduler"], torch.optim.lr_scheduler.OneCycleLR)
    assert policy["interval"] == "step"


def test_build_returns_a_ready_experiment(dataset_root: Path) -> None:
    experiment = build(disk_config(dataset_root))

    assert isinstance(experiment.module, TrainingModule)
    assert isinstance(experiment.data, TrainingData)
    assert isinstance(experiment.trainer, L.Trainer)
    assert [task.name for task in experiment.module.tasks] == ["label"]


def test_facts_inferred_from_data_reach_the_model(dataset_root: Path) -> None:
    """The ordering contract: setup runs before the model, so heads are sized from data."""
    experiment = build(disk_config(dataset_root))

    batch = next(iter(experiment.data.val_dataloader()))
    assert tensor(experiment.module.model.predict(batch).outputs["label"]).shape[1] == 2


def test_trainer_knobs_forward_from_config(dataset_root: Path) -> None:
    experiment = build(disk_config(dataset_root, trainer={"max_epochs": 7}))

    assert experiment.trainer.max_epochs == 7


def test_the_trainers_output_root_is_whatever_config_declared(dataset_root: Path) -> None:
    """One output root for everything a run writes, and config is what names it.

    The shipped group writes `default_root_dir: ${run.directory}`, so nothing here
    fills it in — which is why a config built from a dict, as this one is, has to
    say it. `test_run_directory.py` is what holds the shipped groups to the line.
    """
    config = disk_config(dataset_root, trainer={"max_epochs": 1, "default_root_dir": str(dataset_root)})

    assert build(config).trainer.default_root_dir == str(dataset_root)


class CountingKind(Classification):
    """A kind that counts its constructions — a stand-in for one with a side effect in ``__init__``."""

    built: ClassVar[int] = 0

    def __init__(self) -> None:
        super().__init__()
        CountingKind.built += 1


def test_each_kind_is_built_once_for_the_whole_experiment(dataset_root: Path) -> None:
    """The root resolves a task's kind once and hands it to every package that asks — the data
    pipeline, the tasks, the metrics — so a kind is not three objects that happen to agree."""
    CountingKind.built = 0
    counting = {"_target_": "tests.unit.test_build.test_experiment.CountingKind"}
    config = disk_config(dataset_root, tasks={"label": TASKS["label"] | {"kind": counting}})

    build(config)

    assert CountingKind.built == 1
