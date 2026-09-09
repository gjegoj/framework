"""A batch transform declared in config reaches the run bound to its tasks, and nothing else has to change."""

from __future__ import annotations

from functools import partial
from typing import Any

import lightning as L
import pytest
import torch

from src.build import build_callbacks
from src.callbacks import ApplyBatchTransform
from src.core import Batch, TaskFacts
from src.tasks import Segmentation, Task
from src.training import TrainingModule
from tests.support.configs import paper_config
from tests.support.entities import a_task
from tests.support.fakes import a_composite
from tests.support.lightning import quiet_trainer
from tests.support.narrowing import tensor


def trainer() -> L.Trainer:
    return quiet_trainer(max_epochs=10)


def module(*tasks: Task) -> TrainingModule:
    return TrainingModule(
        model=a_composite(12, classes=3), tasks=tasks, optimizer_factory=partial(torch.optim.SGD, lr=0.1)
    )


def declared(**params: Any) -> dict[str, Any]:
    return {"name": "batch_transform", "transform": {"_target_": "src.transforms.MixUp", **params}}


def test_the_tasks_and_their_class_counts_reach_the_transform() -> None:
    """Asserted through the built callback's effect: nothing else proves they arrived."""
    task = a_task(facts=TaskFacts(num_classes=3))
    batch = Batch(inputs={"image": torch.rand(2, 3, 8, 8)}, targets={"label": torch.tensor([0, 1])})
    built = build_callbacks(paper_config(callbacks=[declared(alpha=0.4)]))
    assert isinstance(built[0], ApplyBatchTransform)

    built[0].setup(trainer(), module(task), stage="fit")
    built[0].on_train_batch_start(trainer(), module(task), batch, 0)

    assert tensor(batch.targets["label"]).shape == (2, 3)  # the three classes the task's facts carry


def test_a_stack_the_tasks_cannot_support_fails_when_the_trainer_sets_it_up() -> None:
    """Before the first batch is the latest honest moment: MixUp cannot serve a mask, and the run is told which."""
    built = build_callbacks(paper_config(callbacks=[declared()]))
    dense = a_task(name="mask", kind=Segmentation(), facts=TaskFacts(num_classes=3))

    with pytest.raises(ValueError, match="mask"):
        built[0].setup(trainer(), module(dense), stage="fit")
