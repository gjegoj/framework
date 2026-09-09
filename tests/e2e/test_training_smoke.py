"""End-to-end smoke: the full chain trains for one epoch on synthetic data.

Table source, schema, profiling, task building, composite model, the single
``TrainingModule``, and Lightning fit/test — every layer participates.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from src.config import MetricConfig
from src.core import (
    Backbone,
    Features,
)
from src.metrics.build import build_metric_sets
from src.models import CompositeModel
from src.tasks import Classification
from src.training import TrainingData, TrainingModule
from tests.support.entities import a_task
from tests.support.lightning import quiet_trainer
from tests.support.tables import in_memory_pipeline


class PointBackbone(Backbone):
    def forward(self, inputs: dict[str, Tensor]) -> Features:
        return Features(streams={"features": inputs["point"]})

    def feature_dims(self) -> Mapping[str, int]:
        return {"features": 2}


def load_point(value: Any) -> Tensor:
    number = float(value)
    return torch.tensor([number, -number])


def shipped_monitors() -> set[str]:
    """Every metric key the shipped callbacks group asks a callback to watch."""
    declared = yaml.safe_load(Path("configs/callbacks/default.yaml").read_text(encoding="utf-8"))
    return {entry["monitor"] for entry in declared["callbacks"] if "monitor" in entry}


def test_one_epoch_of_training_runs_through_every_layer() -> None:
    data_module, facts = in_memory_pipeline(16, input="point", loader=load_point)

    task = a_task(facts=facts["label"])
    metrics = build_metric_sets(
        Classification(), facts=facts["label"], metrics={"accuracy": MetricConfig(name="accuracy")}
    )
    backbone = PointBackbone()
    model = CompositeModel(backbone=backbone, components={task.name: task.kind.components(task, backbone)})
    module = TrainingModule(
        model=model,
        tasks=[task],
        metrics={task.name: metrics},
        optimizer_factory=partial(torch.optim.SGD, lr=0.05),
    )
    data = TrainingData(data_module, batch_size=4)
    trainer = quiet_trainer(devices=1)

    trainer.fit(module, datamodule=data)
    fit_metrics = dict(trainer.callback_metrics)  # trainer.test() resets callback_metrics
    trainer.test(module, datamodule=data)

    assert "train/loss" in fit_metrics
    assert "train/label/ce" in fit_metrics
    # The shipped config's monitor has to be a key this chain really logs. Asserted here
    # rather than in a unit test because only a run knows what the keys are — checking it
    # against the constructor, as the callback unit tests do, cannot tell a real key from
    # an invented one.
    assert shipped_monitors() <= set(fit_metrics)
    assert 0.0 <= fit_metrics["val/label/accuracy"].item() <= 1.0
    assert 0.0 <= trainer.callback_metrics["test/label/accuracy"].item() <= 1.0
