"""The whole chain on one tiny table: a picture on disk becomes a loss, and a gradient comes back.

Every phase so far meets here — encoders, the pixel pipeline declared as a run declares it, the table
module, a backbone, a head sized from both ends, a task's three views, its loss and its metrics. The
wiring is written out the way the composition root will write it, so a seam that does not fit fails
once, with the whole path in view, instead of surfacing a shape error deep in a real run.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from src.config import ComponentConfig, HeadConfig, ModelConfig, PreprocessingConfig, TaskConfig
from src.core import Batch, Stage, require_tensor
from src.data.build import build_data_module, build_preprocessor, build_transforms
from src.losses.build import build_loss
from src.metrics.build import build_metrics
from src.models.build import build_model
from src.tasks.registry import task_registry
from src.tracking.artifacts import Matrix
from src.training import StandardLearner

pytestmark = pytest.mark.e2e

SIZE = [32, 32]
SAMPLES = 8
PIPELINE = ComponentConfig.model_validate(
    {
        "_target_": "src.transforms.AlbumentationsTransform",
        "transforms": [
            {"_target_": "albumentations.Resize", "height": SIZE[0], "width": SIZE[1]},
            {"_target_": "albumentations.Normalize", "mean": [0.5] * 3, "std": [0.5] * 3},
            {"_target_": "albumentations.pytorch.ToTensorV2"},
        ],
    }
)
"""What `configs/transforms/default.yaml` declares, spelled out here rather than composed by Hydra."""


@pytest.fixture(scope="module")
def table(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Eight pictures of two kinds on disk, and the table that names them."""
    root = tmp_path_factory.mktemp("pets")
    rows = []
    for index in range(SAMPLES):
        picture = np.full((12, 10, 3), 20 * index + 30, dtype=np.uint8)
        cv2.imwrite(str(root / f"{index}.png"), picture)
        rows.append({"image_path": f"{index}.png", "species": "cat" if index % 2 else "dog"})
    pd.DataFrame(rows).to_csv(root / "rows.csv", index=False)
    return root / "rows.csv"


@pytest.fixture(scope="module")
def run(table: Path) -> tuple[StandardLearner, MetricCollection, Batch]:
    """Assemble a run from declarations alone, and hand back what a training step needs."""
    declared = TaskConfig.model_validate(
        {"kind": "classification", "target": "species", "classes": {0: "cat", 1: "dog"}}
    )
    kind = task_registry.get(str(declared.kind.name))

    preprocessor = build_preprocessor(
        PreprocessingConfig.model_validate(
            {"name": "standard", "inputs": {"image": {"name": "image", "image_size": SIZE, "root": str(table.parent)}}}
        ),
        {"species": declared},
        {"species": ComponentConfig(name=kind.default_target_encoder)},
    )
    data = build_data_module(
        ComponentConfig.model_validate(
            {
                "name": "table",
                "source": str(table),
                "inputs": {"image": {"column": "image_path"}},
                "split": {"fractions": {"train": 0.5, "val": 0.5}, "seed": 0},
            }
        ),
        preprocessor=preprocessor,
        targets={"species": "species"},
        transforms=build_transforms({Stage.TRAIN: PIPELINE, Stage.VAL: PIPELINE}, preprocessor.geometries),
    )
    data.setup(("train", "val"))
    data.fit_preprocessing("train")

    info = data.info.targets["species"]
    task = kind("species", info)
    model = build_model(
        ModelConfig.model_validate(
            {"name": "composite", "backbone": {"name": "timm", "model_name": "resnet18", "pretrained": False}}
        ),
        heads={"species": HeadConfig.model_validate(task.default_head)},
        outputs={"species": task.output_shape(info)},
    )
    learner = StandardLearner(model, {"species": task}, {"species": build_loss(task.default_loss, info)})
    loader = DataLoader(data.dataset("train"), batch_size=4, collate_fn=preprocessor.collate)
    return learner, build_metrics(type(task).default_metrics, task.metric_kwargs()), next(iter(loader))


def test_a_picture_on_disk_reaches_a_loss_and_a_gradient_comes_back(
    run: tuple[StandardLearner, MetricCollection, Batch],
) -> None:
    learner, _, batch = run

    step = learner.step(batch)
    assert step.loss is not None
    step.loss.total.backward()

    assert require_tensor(batch.inputs["image"], name="image").shape == (4, 3, *SIZE)
    assert step.loss.total.ndim == 0 and torch.isfinite(step.loss.total)
    assert set(step.loss.losses) == {"species/cross_entropy"}
    encoder = next(value for name, value in learner.named_parameters() if name.startswith("model.backbone."))
    assert encoder.grad is not None and torch.any(encoder.grad != 0)


def test_the_metrics_the_task_declares_score_the_step_it_produced(
    run: tuple[StandardLearner, MetricCollection, Batch],
) -> None:
    """The predictions a step answers with are what a metric reads: nothing in between reshapes them."""
    learner, metrics, batch = run

    step = learner.step(batch)
    metrics.update(step.predictions["species"], step.targets["species"])
    computed = metrics.compute()

    assert set(computed) == {"f1", "precision", "recall", "confusion_matrix"}
    assert isinstance(computed["confusion_matrix"], Matrix)
    assert computed["f1"].shape == (2,)
