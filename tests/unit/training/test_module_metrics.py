"""Epoch-end metric reporting: values leave the module by their geometry."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import Tensor
from torchmetrics import F1Score

from src.core import DataProfile, Objective, Stage, Task, Topology
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
from src.metrics import WrappedMetricSet
from src.models import CompositeModel, LinearHead, TaskComponents
from src.tasks.adapters import as_class_indices
from src.training import TrainingData, TrainingModule
from tests.support.fakes import FlattenBackbone
from tests.support.lightning import quiet_trainer


def load_pair(value: Any) -> Tensor:
    return torch.tensor([float(value), 1.0])


def test_a_per_class_metric_lands_as_mean_plus_named_leaves(tmp_path: Any) -> None:
    """The preset default is `average: none`; a run must log it, not crash on a vector."""
    per_class = WrappedMetricSet({"f1": F1Score(task="multiclass", num_classes=2, average="none")})
    task = Task(
        name="label",
        topology=Topology.GLOBAL,
        objective=Objective.MULTICLASS,
        metrics={Stage.TRAIN: per_class},
        class_names=["cat", "dog"],
    )
    model = CompositeModel(
        backbone=FlattenBackbone(dim=2),
        components={
            "label": TaskComponents(
                head=LinearHead(2, 2),
                criterion=CrossEntropyCriterion(),
                activation=lambda logits: logits,
                target_adapter=as_class_indices,
            )
        },
    )
    module = TrainingModule(model=model, tasks=[task], optimizer_factory=partial(torch.optim.SGD, lr=0.1))
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
    trainer = quiet_trainer(limit_val_batches=0, default_root_dir=tmp_path)

    trainer.fit(module, datamodule=TrainingData(data_module, batch_size=2))

    logged = set(trainer.callback_metrics)
    assert {"train/label/f1/mean", "train/label/f1/cat", "train/label/f1/dog"} <= logged


def test_the_training_module_depends_on_core_alone() -> None:
    """Routing is core policy; a capability import here would be a boundary leak."""
    import src.training.module

    source = Path(src.training.module.__file__).read_text()
    assert "from src.metrics" not in source


def test_the_module_reports_its_metrics_directions_under_logged_keys() -> None:
    """Consumers (the progress bar) rank values without re-deriving semantics from names."""
    from torchmetrics import MeanAbsoluteError

    from src.core.ports import MetricDirectionProvider
    from src.models import LinearHead, TaskComponents

    task = Task(
        name="label",
        topology=Topology.GLOBAL,
        objective=Objective.MULTICLASS,
        metrics={
            Stage.TRAIN: WrappedMetricSet({"f1": F1Score(task="multiclass", num_classes=2)}),
            Stage.VAL: WrappedMetricSet({"mae": MeanAbsoluteError()}),
        },
    )
    module = TrainingModule(
        model=CompositeModel(
            backbone=FlattenBackbone(dim=2),
            components={
                "label": TaskComponents(
                    head=LinearHead(2, 2),
                    criterion=CrossEntropyCriterion(),
                    activation=lambda logits: logits,
                    target_adapter=as_class_indices,
                )
            },
        ),
        tasks=[task],
        optimizer_factory=partial(torch.optim.SGD, lr=0.1),
    )

    assert isinstance(module, MetricDirectionProvider)
    directions = module.metric_directions()
    assert directions["train/label/f1"] is True
    assert directions["val/label/mae"] is False
