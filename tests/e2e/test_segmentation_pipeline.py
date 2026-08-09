"""End-to-end: image and mask files through joint augmentation into a dense loss.

The acceptance test for the raw-encoder design — a mask must survive loading,
share the image's geometry through augmentation, and arrive at the criterion
as long class indices.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import albumentations as A
import pytest
import torch
from albumentations.pytorch import ToTensorV2
from torch import Tensor, nn

from src.core import Backbone, DataProfile, Features, Objective, Stage, Stream, Task, Topology
from src.data import (
    CsvSource,
    DataSchema,
    ImageLoader,
    InputColumn,
    MaskTargetEncoder,
    TableDataModule,
    TargetColumn,
    collate_samples,
    random_split,
)
from src.models import CompositeModel
from src.tasks import build_task_components
from src.transforms import AlbumentationsTransform
from tests.support.datasets import write_dataset
from tests.support.narrowing import tensor

CLASSES = 3
SAMPLES = 8


class TinyDecoderBackbone(Backbone):
    """A one-conv stand-in exposing the dense stream a segmentation head reads."""

    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self._convolution = nn.Conv2d(3, width, kernel_size=3, padding=1)
        self._width = width

    def forward(self, inputs: dict[str, Tensor]) -> Features:
        return Features(streams={Stream.DECODER: self._convolution(inputs["image"])})

    def feature_dims(self) -> Mapping[str, int]:
        return {Stream.DECODER: self._width}


@pytest.mark.e2e
def test_masks_travel_from_files_through_augmentation_into_a_dense_loss(tmp_path: Path) -> None:
    table_path = write_dataset(tmp_path, rows=SAMPLES, side=12, masks=True, mask_classes=CLASSES)
    data_module = TableDataModule(
        source=CsvSource(table_path),
        schema=DataSchema(
            inputs={"image": InputColumn(column="image", loader=ImageLoader(root=tmp_path))},
            targets={
                "mask": TargetColumn(column="mask", encoder=MaskTargetEncoder(num_classes=CLASSES, root=tmp_path))
            },
        ),
        splitter=random_split({Stage.TRAIN: 0.5, Stage.VAL: 0.25, Stage.TEST: 0.25}, seed=42),
        transforms=dict.fromkeys(
            Stage,
            AlbumentationsTransform(
                [A.Resize(height=8, width=8), A.Normalize(), ToTensorV2()],
                spatial_targets=["mask"],
            ),
        ),
    )
    profile = DataProfile()
    data_module.setup(profile)

    dataset = data_module.dataset(Stage.TRAIN)
    batch = collate_samples([dataset[index] for index in range(len(dataset))])

    assert batch.inputs["image"].shape == (4, 3, 8, 8)
    assert tensor(batch.targets["mask"]).shape == (4, 8, 8)
    assert profile.require_num_classes("mask") == CLASSES

    task = Task(name="mask", topology=Topology.DENSE, objective=Objective.MULTICLASS, metrics={})
    backbone = TinyDecoderBackbone()
    model = CompositeModel(
        backbone=backbone,
        components={task.name: build_task_components(task, profile, backbone)},
    )

    loss, prediction, targets = model.step(batch)

    assert set(loss.parts) == {"mask/ce"}
    assert loss.total.requires_grad
    assert tensor(prediction.outputs["mask"]).shape == (4, CLASSES, 8, 8)
    assert tensor(targets["mask"]).dtype == torch.long
