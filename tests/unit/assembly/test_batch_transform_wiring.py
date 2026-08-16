"""A batch transform gets what only assembly knows, and nothing else has to change."""

from __future__ import annotations

from typing import Any, cast

import lightning as L
import pytest
import torch

from src.assembly.callbacks import build_callbacks
from src.callbacks import ApplyBatchTransform
from src.core import Batch, DataProfile, Objective, OutputTopology, TargetFacts, Task
from tests.support.configs import paper_config
from tests.support.lightning import quiet_trainer
from tests.support.narrowing import tensor


def trainer() -> L.Trainer:
    return quiet_trainer(max_epochs=10)


def classification() -> tuple[list[Task], DataProfile]:
    profile = DataProfile()
    profile.record("label", TargetFacts(num_classes=3))
    return [
        Task(name="label", output_topology=OutputTopology.GLOBAL, objective=Objective.MULTICLASS, metrics={})
    ], profile


def declared(**params: Any) -> dict[str, Any]:
    return {"name": "batch_transform", "transform": {"_target_": "src.transforms.MixUp", **params}}


def test_the_tasks_and_their_class_counts_reach_the_transform() -> None:
    """Asserted through the built callback's effect: nothing else proves they arrived."""
    tasks, profile = classification()
    batch = Batch(inputs={"image": torch.rand(2, 3, 8, 8)}, targets={"label": torch.tensor([0, 1])})

    built = build_callbacks(paper_config(callbacks=[declared(alpha=0.4)]), tasks, profile)
    built[0].on_train_batch_start(trainer(), cast("L.LightningModule", None), batch, 0)

    assert isinstance(built[0], ApplyBatchTransform)
    assert tensor(batch.targets["label"]).shape == (2, 3)  # the three classes the profile knew about


def test_callbacks_that_do_not_want_them_are_unaffected() -> None:
    """The derived values are offered, not forced — that is the whole seam."""
    tasks, profile = classification()

    built = build_callbacks(paper_config(callbacks=[{"name": "lr_monitor"}]), tasks, profile)

    assert len(built) == 1


def test_a_stack_the_tasks_cannot_support_fails_at_assembly() -> None:
    """An hour into training is the wrong time to learn that MixUp cannot serve a mask."""
    profile = DataProfile()
    profile.record("mask", TargetFacts(num_classes=3))
    dense = [Task(name="mask", output_topology=OutputTopology.DENSE, objective=Objective.MULTICLASS, metrics={})]

    with pytest.raises(ValueError, match="mask"):
        build_callbacks(paper_config(callbacks=[declared()]), dense, profile)
