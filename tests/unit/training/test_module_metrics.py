"""Epoch-end metric reporting: values leave the module by their geometry."""

from __future__ import annotations

from functools import partial
from typing import Any

import torch
from torchmetrics import F1Score

from src.core import Stage
from src.metrics import WrappedMetricSet
from src.training import TrainingData, TrainingModule
from tests.support.entities import a_task
from tests.support.fakes import a_composite
from tests.support.lightning import quiet_trainer
from tests.support.tables import in_memory_pipeline


def test_a_per_class_metric_lands_as_mean_plus_named_leaves(tmp_path: Any) -> None:
    """The preset default is `average: none`; a run must log it, not crash on a vector."""
    per_class = WrappedMetricSet({"f1": F1Score(task="multiclass", num_classes=2, average="none")})
    task = a_task(class_names=["cat", "dog"])
    module = TrainingModule(
        model=a_composite(2),
        tasks=[task],
        metrics={"label": {Stage.TRAIN: per_class}},
        optimizer_factory=partial(torch.optim.SGD, lr=0.1),
    )
    data_module, _ = in_memory_pipeline()
    trainer = quiet_trainer(limit_val_batches=0, default_root_dir=tmp_path)

    trainer.fit(module, datamodule=TrainingData(data_module, batch_size=2))

    logged = set(trainer.callback_metrics)
    assert {"train/label/f1/mean", "train/label/f1/cat", "train/label/f1/dog"} <= logged


def test_the_module_reports_its_metrics_directions_under_logged_keys() -> None:
    """Consumers (the progress bar) rank values without re-deriving semantics from names."""
    from torchmetrics import MeanAbsoluteError

    from src.training.ports import DeclaresMetricDirections

    module = TrainingModule(
        model=a_composite(2),
        tasks=[a_task()],
        metrics={
            "label": {
                Stage.TRAIN: WrappedMetricSet({"f1": F1Score(task="multiclass", num_classes=2)}),
                Stage.VAL: WrappedMetricSet({"mae": MeanAbsoluteError()}),
            }
        },
        optimizer_factory=partial(torch.optim.SGD, lr=0.1),
    )

    assert isinstance(module, DeclaresMetricDirections)
    directions = module.metric_directions()
    assert directions["train/label/f1"] is True
    assert directions["val/label/mae"] is False
