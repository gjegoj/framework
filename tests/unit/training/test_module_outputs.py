"""What the module does when a model answers for fewer tasks than the run declared."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import pytest
import torch
from torch import nn
from torch.nn import functional
from torchmetrics import Accuracy

from src.core import Loss, Model, Prediction, Stage, StepResult
from src.metrics import WrappedMetricSet
from src.training import TrainingData, TrainingModule
from tests.support.entities import a_task
from tests.support.lightning import quiet_trainer
from tests.support.tables import in_memory_pipeline

if TYPE_CHECKING:
    from pathlib import Path

    from src.core import Batch


class SilentModel(Model):
    """A model that arrives whole, trains fine, and never names the task in its outputs — an integration slip."""

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(2, 2)

    def step(self, batch: Batch) -> StepResult:
        logits = self.classifier(batch.inputs["image"])
        target = batch.targets["label"]
        assert isinstance(target, torch.Tensor)
        loss = Loss.part("ce", functional.cross_entropy(logits, target)).scoped("label")
        return StepResult(loss=loss, prediction=self.predict(batch), targets=dict(batch.targets))

    def predict(self, batch: Batch) -> Prediction:
        return Prediction(outputs={}, logits={})


def test_a_task_whose_metrics_saw_no_output_in_an_evaluation_epoch_is_refused_by_name(tmp_path: Path) -> None:
    """A training step may legitimately produce nothing to evaluate (a head that only decodes in eval
    mode), so a missing output is skipped there — but an evaluation epoch that ends without one for a
    task with metrics would report a metric computed on nothing, and says so instead."""
    module = TrainingModule(
        model=SilentModel(),
        tasks=[a_task()],
        metrics={"label": {Stage.VAL: WrappedMetricSet({"accuracy": Accuracy(task="multiclass", num_classes=2)})}},
        optimizer_factory=partial(torch.optim.SGD, lr=0.1),
    )
    data_module, _ = in_memory_pipeline()

    with pytest.raises(ValueError, match=r"Task 'label'.*val"):
        quiet_trainer(default_root_dir=tmp_path).fit(module, datamodule=TrainingData(data_module, batch_size=2))
