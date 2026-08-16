"""``TrainingModule`` wiring: parameters, metric registration, optimizer factory."""

from __future__ import annotations

from functools import partial
from typing import Any

import pandas as pd
import torch
from lightning.pytorch.utilities.types import LRSchedulerConfigType
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR

from src.core import Batch, DataProfile, Loss, Objective, OutputTopology, Prediction, Stage, StepResult, Task
from src.core.ports import Model
from src.data import (
    DataSchema,
    InMemorySource,
    InputColumn,
    LabelTargetEncoder,
    TableDataModule,
    TargetColumn,
    random_split,
)
from src.losses import CrossEntropyCriterion
from src.models import CompositeModel, LinearHead, TaskComponents
from src.training import FitProfile, SchedulerFactory, TrainingData, TrainingModule
from tests.support.entities import as_is
from tests.support.fakes import CountingMetricSet, FlattenBackbone
from tests.support.lightning import quiet_trainer


def make_module(scheduler_factory: SchedulerFactory | None = None) -> tuple[TrainingModule, CountingMetricSet]:
    metrics = CountingMetricSet()
    task = Task(
        name="label",
        output_topology=OutputTopology.GLOBAL,
        objective=Objective.MULTICLASS,
        metrics={Stage.TRAIN: metrics},
    )
    model = CompositeModel(
        backbone=FlattenBackbone(dim=2),
        components={
            "label": TaskComponents(
                head=LinearHead(2, 2),
                criterion=CrossEntropyCriterion(),
                activation=lambda logits: logits,
                target_adapter=as_is,
            )
        },
    )
    module = TrainingModule(
        model=model,
        tasks=[task],
        optimizer_factory=partial(torch.optim.SGD, lr=0.1),
        scheduler_factory=scheduler_factory,
    )
    return module, metrics


def load_pair(value: Any) -> Tensor:
    return torch.tensor([float(value), 1.0])


def make_training_data() -> TrainingData:
    table = pd.DataFrame({"x": [float(index) for index in range(8)], "label": ["cat", "dog"] * 4})
    data_module = TableDataModule(
        source=InMemorySource(table),
        schema=DataSchema(
            inputs={"image": InputColumn(column="x", loader=load_pair)},
            targets={"label": TargetColumn(column="label", encoder=LabelTargetEncoder())},
        ),
        splitter=random_split({Stage.TRAIN: 0.5, Stage.VAL: 0.25, Stage.TEST: 0.25}, seed=42),
    )
    data_module.setup(DataProfile())
    return TrainingData(data_module, batch_size=2)


def test_module_exposes_the_model_parameters() -> None:
    module, _ = make_module()

    parameter_names = dict(module.named_parameters())

    assert any("heads.label" in name for name in parameter_names)


def test_metrics_are_registered_for_device_movement() -> None:
    module, metrics = make_module()

    assert any(child is metrics for child in module.modules())


def test_configure_optimizers_uses_the_factory() -> None:
    module, _ = make_module()

    optimizer = module.configure_optimizers()

    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["lr"] == 0.1


def test_a_scheduler_factory_receives_facts_only_the_trainer_knows() -> None:
    """The module owns the trainer, so it is the module that supplies fit-time facts."""
    seen: list[FitProfile] = []

    def scheduler_factory(optimizer: Optimizer, profile: FitProfile) -> LRSchedulerConfigType:
        seen.append(profile)
        return {"scheduler": StepLR(optimizer, step_size=1), "interval": "epoch"}

    module, _ = make_module(scheduler_factory=scheduler_factory)
    trainer = quiet_trainer(fast_dev_run=True, devices=1)

    trainer.fit(module, datamodule=make_training_data())

    assert len(seen) == 1
    assert seen[0].total_steps > 0
    assert seen[0].steps_per_epoch > 0
    assert trainer.lr_scheduler_configs != []


def test_a_task_a_step_produced_nothing_for_contributes_no_metric() -> None:
    """Absent is a real answer, and it is not the same answer as empty.

    A vendor head assembles its decodable output only in eval mode, so a training step
    genuinely has no prediction to judge. Handing the metric a fabricated blank would
    make it report a score for a measurement nobody took — a zero that reads as a broken
    model rather than as an epoch with no train-stage numbers.
    """
    module, metrics = make_module()
    silent = Batch(inputs={"image": torch.zeros(2, 2)}, targets={"label": torch.zeros(2, dtype=torch.long)})
    module.model = _ModelWithNoPrediction()

    module.training_step(silent, 0)

    assert metrics.seen == 0


class _ModelWithNoPrediction(Model):
    """A family that computes a loss and produces no prediction for the step's stage."""

    def step(self, batch: Batch) -> StepResult:
        return StepResult(
            loss=Loss.part("box", torch.zeros((), requires_grad=True)),
            prediction=Prediction(outputs={}),
            targets=dict(batch.targets),
        )

    def predict(self, batch: Batch) -> Prediction:
        return Prediction(outputs={})
