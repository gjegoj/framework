"""``TrainingModule`` wiring: parameters, metric registration, optimizer factory."""

from __future__ import annotations

from functools import partial

import torch
from lightning.pytorch.utilities.types import LRSchedulerConfigType
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR

from src.core import Batch, Loss, Prediction, Stage, StepResult
from src.core.ports import Model
from src.training import FitProfile, SchedulerFactory, TrainingData, TrainingModule
from tests.support.entities import a_task
from tests.support.fakes import CountingMetricSet, a_composite
from tests.support.lightning import quiet_trainer
from tests.support.tables import in_memory_pipeline


def make_module(scheduler_factory: SchedulerFactory | None = None) -> tuple[TrainingModule, CountingMetricSet]:
    metrics = CountingMetricSet()
    module = TrainingModule(
        model=a_composite(2),
        tasks=[a_task()],
        metrics={"label": {Stage.TRAIN: metrics}},
        optimizer_factory=partial(torch.optim.SGD, lr=0.1),
        scheduler_factory=scheduler_factory,
    )
    return module, metrics


def make_training_data() -> TrainingData:
    data_module, _ = in_memory_pipeline()
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

    A detection head assembles its decodable output only in eval mode, so a training step
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


def test_the_module_publishes_its_tasks_for_callbacks() -> None:
    """Lightning hands every callback the module; a callback that needs the tasks reads them there."""
    module, _ = make_module()

    assert [task.name for task in module.tasks] == ["label"]
