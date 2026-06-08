"""Unit and smoke tests for the training layer."""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import lightning as L
import numpy as np
import pandas as pd
import pytest
import torch
from albumentations.pytorch import ToTensorV2

from src.core.entities import LossResult
from src.core.enums import Stage
from src.core.runtime import RuntimeContext
from src.data import (
    AlbumentationsTransform,
    CsvDataSource,
    DataModule,
    LabelIndexCodec,
    TargetBinding,
)
from src.models import build_composite_model
from src.models.backbones import TimmBackbone
from src.tasks import classification
from src.training import (
    LitDataModule,
    LitModule,
    OptimizerBuilder,
    WeightedSumAggregator,
)

# ---------------------------------------------------------------- helpers


def _make_transform(
    size: tuple[int, int] = (32, 32),
    spatial: list[str] | None = None,
) -> AlbumentationsTransform:
    h, w = size
    compose = A.Compose([A.Resize(h, w, mask_interpolation=0), A.Normalize(), ToTensorV2()])
    return AlbumentationsTransform(compose, spatial_targets=spatial)


def _loss(value: float, name: str = "ce") -> LossResult:
    t = torch.tensor(value)
    return LossResult(total=t, components={name: t})


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    """15 synthetic 32x32 images, 3 classes."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rng = np.random.default_rng(1)
    rows = []
    labels = ["cat", "dog", "cow"]
    for i in range(15):
        arr = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        p = image_dir / f"{i}.jpg"
        cv2.imwrite(str(p), arr)
        rows.append({"image_path": str(p), "label": labels[i % 3]})
    csv = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


# ---------------------------------------------------------- aggregator


class TestWeightedSumAggregator:
    def test_single_task_total(self) -> None:
        agg = WeightedSumAggregator()
        result = agg.combine({"a": _loss(2.0)}, {"a": 1.0})
        assert result.total.item() == pytest.approx(2.0)

    def test_weighted_sum(self) -> None:
        agg = WeightedSumAggregator()
        result = agg.combine({"a": _loss(2.0), "b": _loss(4.0)}, {"a": 0.5, "b": 0.5})
        assert result.total.item() == pytest.approx(3.0)

    def test_components_namespaced(self) -> None:
        agg = WeightedSumAggregator()
        result = agg.combine({"task": _loss(1.0, "cross_entropy")}, {"task": 1.0})
        assert "task/cross_entropy" in result.components


# --------------------------------------------------------- optimizer


class TestOptimizerBuilder:
    def test_single_group_without_overrides(self) -> None:
        model = TimmBackbone("resnet18", pretrained=False)
        opt = OptimizerBuilder(base_lr=1e-3).build(model)
        assert len(opt.param_groups) == 1
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)

    def test_per_head_lr_creates_two_groups(self) -> None:
        backbone = TimmBackbone("resnet18", pretrained=False)
        from src.core.entities import HeadSpec
        from src.models import build_composite_model

        model = build_composite_model(backbone, {"label": HeadSpec(kind="linear", out_features=3)})

        opt = OptimizerBuilder(base_lr=1e-3).build(model, task_lr_overrides={"label": 1e-4})
        assert len(opt.param_groups) == 2
        lrs = {g["name"]: g["lr"] for g in opt.param_groups}
        assert lrs["backbone"] == pytest.approx(1e-3)
        assert lrs["head/label"] == pytest.approx(1e-4)

    def test_no_param_overlap_between_groups(self) -> None:
        backbone = TimmBackbone("resnet18", pretrained=False)
        from src.core.entities import HeadSpec

        model = build_composite_model(backbone, {"label": HeadSpec(kind="linear", out_features=3)})

        opt = OptimizerBuilder(base_lr=1e-3).build(model, task_lr_overrides={"label": 1e-4})
        all_ids: list[int] = []
        for group in opt.param_groups:
            all_ids.extend(id(p) for p in group["params"])
        assert len(all_ids) == len(set(all_ids)), "params overlap between groups"

    def test_from_name_builds_sgd_with_extras(self) -> None:
        model = TimmBackbone("resnet18", pretrained=False)
        builder = OptimizerBuilder.from_name(
            "sgd",
            base_lr=1e-2,
            base_weight_decay=1e-4,
            extra_kwargs={"momentum": 0.9, "nesterov": True},
        )
        opt = builder.build(model)
        assert isinstance(opt, torch.optim.SGD)
        assert opt.param_groups[0]["momentum"] == pytest.approx(0.9)
        assert opt.param_groups[0]["nesterov"] is True

    def test_build_optimizer_builder_reads_name_and_extras(self) -> None:
        from src.composition.wiring import build_optimizer_builder
        from src.config.schema import OptimizerConfig

        cfg = OptimizerConfig(name="sgd", lr=1e-2, weight_decay=1e-4, momentum=0.8)
        opt = build_optimizer_builder(cfg).build(TimmBackbone("resnet18", pretrained=False))
        assert isinstance(opt, torch.optim.SGD)
        assert opt.param_groups[0]["momentum"] == pytest.approx(0.8)


# ------------------------------------------------ LitModule smoke


class TestLitModuleSmoke:
    """Single-epoch CPU smoke test — validates the full step loop works."""

    def test_fit_one_epoch(self, csv_path: Path) -> None:
        runtime = RuntimeContext()
        transforms = {s: _make_transform((32, 32)) for s in Stage}
        plain_dm = DataModule(
            target_bindings=[
                TargetBinding("label", "label", LabelIndexCodec(class_mapping={0: "cat", 1: "cow", 2: "dog"}))
            ],
            inputs_config="image_path",
            transforms=transforms,
            runtime=runtime,
            batch_size=4,
            seed=0,
            source=CsvDataSource(str(csv_path)),
            split={Stage.TRAIN: 0.6, Stage.VAL: 0.2, Stage.TEST: 0.2},
            drop_last=True,  # 15 samples × 0.6 = 9 train → last batch of 1 breaks BN
        )
        plain_dm.setup()

        task = classification("label", num_classes=runtime.num_classes["label"])
        backbone = TimmBackbone("resnet18", pretrained=False)
        model = build_composite_model(backbone, {"label": task.head_spec})

        lit_module = LitModule(
            model=model,
            tasks=[task],
            optimizer_builder=OptimizerBuilder(base_lr=1e-3),
        )
        lit_dm = LitDataModule(plain_dm)

        trainer = L.Trainer(
            max_epochs=1,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(lit_module, lit_dm)

        # val accuracy should be logged
        assert "label/f1/val/mean" in trainer.logged_metrics


class TestSegmentationSmoke:
    """End-to-end DENSE segmentation: smp backbone + mask pipeline + trainer.fit."""

    @pytest.fixture
    def seg_csv(self, tmp_path: Path) -> Path:
        img_dir = tmp_path / "img"
        msk_dir = tmp_path / "msk"
        img_dir.mkdir()
        msk_dir.mkdir()
        rng = np.random.default_rng(3)
        rows = []
        for i in range(12):
            cv2.imwrite(str(img_dir / f"{i}.jpg"), rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
            cv2.imwrite(str(msk_dir / f"{i}.png"), rng.integers(0, 3, (64, 64), dtype=np.uint8))
            rows.append({"image_path": str(img_dir / f"{i}.jpg"), "mask_path": str(msk_dir / f"{i}.png")})
        csv = tmp_path / "seg.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        return csv

    def test_fit_one_epoch_segmentation(self, seg_csv: Path) -> None:
        from src.data import MaskCodec
        from src.models.backbones import SmpBackbone
        from src.tasks import segmentation

        runtime = RuntimeContext()
        transforms = {s: _make_transform((64, 64), spatial=["mask"]) for s in Stage}
        plain_dm = DataModule(
            target_bindings=[TargetBinding("mask", "mask_path", MaskCodec())],
            inputs_config="image_path",
            transforms=transforms,
            runtime=runtime,
            batch_size=4,
            seed=0,
            source=CsvDataSource(str(seg_csv)),
            split={Stage.TRAIN: 0.6, Stage.VAL: 0.2, Stage.TEST: 0.2},
        )
        plain_dm.setup()

        task = segmentation("mask", num_classes=3)  # explicit class count
        backbone = SmpBackbone(name="unet", encoder_name="resnet18", pretrained=False)
        model = build_composite_model(backbone, {"mask": task.head_spec})

        lit_module = LitModule(model=model, tasks=[task], optimizer_builder=OptimizerBuilder(base_lr=1e-3))
        trainer = L.Trainer(
            max_epochs=1,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(lit_module, LitDataModule(plain_dm))

        assert "mask/iou/val/mean" in trainer.logged_metrics
