"""CPU training smokes: classification, embedding, ArcFace embedder, segmentation, criterion schedule."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import cast

import cv2
import lightning as L
import numpy as np
import pandas as pd
import pytest
import torch

from src.callbacks.criterion_schedule import CriterionScheduleCallback
from src.core.enums import Stage
from src.core.runtime import RuntimeContext
from src.data import (
    CsvDataSource,
    DataLoaderOptions,
    DataModule,
    LabelEncoder,
    TargetBinding,
)
from src.losses.classification import FocalLoss
from src.losses.registry import criteria
from src.models import build_composite_model
from src.models.backbones import TimmBackbone
from src.tasks import classification
from src.training import (
    LitDataModule,
    LitModule,
    OptimizerBuilder,
)
from tests.support.builders import make_transform as _make_transform


@pytest.fixture
def csv_path(make_image_csv: Callable[..., Path]) -> Path:
    """15 synthetic 32x32 images, 3 classes."""
    return make_image_csv(count=15, size=32, seed=1)


class TestLitModuleSmoke:
    """Single-epoch CPU smoke test — validates the full step loop works."""

    def test_fit_one_epoch(self, csv_path: Path) -> None:
        runtime = RuntimeContext()
        transforms = {s: _make_transform((32, 32)) for s in Stage}
        plain_dm = DataModule(
            target_bindings=[
                TargetBinding("label", "label", LabelEncoder(class_mapping={0: "cat", 1: "cow", 2: "dog"}))
            ],
            inputs_config="image_path",
            transforms=transforms,
            runtime=runtime,
            batch_size=4,
            seed=0,
            source=CsvDataSource(str(csv_path)),
            split={Stage.TRAIN: 0.6, Stage.VAL: 0.2, Stage.TEST: 0.2},
            dataloader_options=DataLoaderOptions(drop_last=True),  # 9 train samples → last batch of 1 breaks BN
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


class TestEmbeddingSmoke:
    """End-to-end classification on precomputed .npy embeddings (modality axis)."""

    @pytest.fixture
    def emb_csv(self, tmp_path: Path) -> Path:
        """15 synthetic 16-dim embedding vectors as .npy files, 3 classes."""
        emb_dir = tmp_path / "emb"
        emb_dir.mkdir()
        rng = np.random.default_rng(2)
        labels = ["cat", "dog", "cow"]
        rows = []
        for i in range(15):
            path = emb_dir / f"{i}.npy"
            np.save(path, rng.standard_normal(16).astype(np.float32))
            rows.append({"emb_path": str(path), "label": labels[i % 3]})
        csv = tmp_path / "emb.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        return csv

    def test_fit_one_epoch_embeddings(self, emb_csv: Path) -> None:
        from src.models.backbones import EmbeddingBackbone
        from src.transforms.sample import IdentityTransform

        runtime = RuntimeContext()
        transforms = {s: IdentityTransform() for s in Stage}
        plain_dm = DataModule(
            target_bindings=[
                TargetBinding("label", "label", LabelEncoder(class_mapping={0: "cat", 1: "cow", 2: "dog"}))
            ],
            inputs_config={"embedding": "emb_path"},  # .npy paths → "embedding" loader auto-detected
            transforms=transforms,
            runtime=runtime,
            batch_size=4,
            seed=0,
            source=CsvDataSource(str(emb_csv)),
            split={Stage.TRAIN: 0.6, Stage.VAL: 0.2, Stage.TEST: 0.2},
            dataloader_options=DataLoaderOptions(drop_last=True),
        )
        plain_dm.setup()

        task = classification("label", num_classes=runtime.num_classes["label"])
        backbone = EmbeddingBackbone(embedding_dim=16, input_key="embedding")
        model = build_composite_model(backbone, {"label": task.head_spec})

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

        assert "label/f1/val/mean" in trainer.logged_metrics


class TestArcFaceEmbeddingSmoke:
    """End-to-end GLOBAL x METRIC proxy classification (arcface_embedding preset)."""

    @pytest.fixture
    def emb_csv(self, tmp_path: Path) -> Path:
        """15 synthetic 8-dim embedding vectors as .npy files, 3 integer classes."""
        emb_dir = tmp_path / "emb"
        emb_dir.mkdir()
        rng = np.random.default_rng(4)
        labels = torch.randint(0, 3, (15,), generator=torch.Generator().manual_seed(4))
        rows = []
        for i in range(15):
            path = emb_dir / f"{i}.npy"
            np.save(path, rng.standard_normal(8).astype(np.float32))
            rows.append({"emb_path": str(path), "label": int(labels[i])})
        csv = tmp_path / "emb.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        return csv

    def test_fit_one_epoch_arcface_embedder(self, emb_csv: Path) -> None:
        from src.losses.angular import ProxyAngularCriterion
        from src.models.backbones import EmbeddingBackbone
        from src.tasks.presets import task_presets
        from src.transforms.sample import IdentityTransform

        runtime = RuntimeContext()
        transforms = {s: IdentityTransform() for s in Stage}
        plain_dm = DataModule(
            target_bindings=[TargetBinding("embed", "label", LabelEncoder(class_mapping={0: "0", 1: "1", 2: "2"}))],
            inputs_config={"embedding": "emb_path"},  # .npy paths → "embedding" loader auto-detected
            transforms=transforms,
            runtime=runtime,
            batch_size=4,
            seed=0,
            source=CsvDataSource(str(emb_csv)),
            split={Stage.TRAIN: 0.6, Stage.VAL: 0.2, Stage.TEST: 0.2},
            dataloader_options=DataLoaderOptions(drop_last=True),
        )
        plain_dm.setup()

        task = task_presets.create("arcface_embedding")("embed", num_classes=8, class_count=3)
        backbone = EmbeddingBackbone(embedding_dim=8, input_key="embedding")
        model = build_composite_model(backbone, {"embed": task.head_spec})

        lit_module = LitModule(model=model, tasks=[task], optimizer_builder=OptimizerBuilder(base_lr=1e-3))
        trainer = L.Trainer(
            max_epochs=1,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        criterion = cast(ProxyAngularCriterion, task.criterion)
        prototypes_before = criterion.prototypes.detach().clone()
        trainer.fit(lit_module, LitDataModule(plain_dm))
        prototypes_after = criterion.prototypes.detach().clone()

        assert not torch.allclose(prototypes_before, prototypes_after)


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
        from src.data import MaskEncoder
        from src.models.backbones import SmpBackbone
        from src.tasks import segmentation

        runtime = RuntimeContext()
        transforms = {s: _make_transform((64, 64), spatial=["mask"]) for s in Stage}
        plain_dm = DataModule(
            target_bindings=[TargetBinding("mask", "mask_path", MaskEncoder())],
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


class TestCriterionScheduleSmoke:
    """3-epoch CPU fit with FocalLoss gamma annealed 2.0 → 0.5 by the callback."""

    def test_gamma_annealed_over_fit(self, csv_path: Path) -> None:
        runtime = RuntimeContext()
        transforms = {s: _make_transform((32, 32)) for s in Stage}
        plain_dm = DataModule(
            target_bindings=[
                TargetBinding("label", "label", LabelEncoder(class_mapping={0: "cat", 1: "cow", 2: "dog"}))
            ],
            inputs_config="image_path",
            transforms=transforms,
            runtime=runtime,
            batch_size=4,
            seed=0,
            source=CsvDataSource(str(csv_path)),
            split={Stage.TRAIN: 0.6, Stage.VAL: 0.2, Stage.TEST: 0.2},
            dataloader_options=DataLoaderOptions(drop_last=True),
        )
        plain_dm.setup()

        task = classification("label", num_classes=runtime.num_classes["label"])
        task = dataclasses.replace(task, criterion=criteria.create("focal"))
        backbone = TimmBackbone("resnet18", pretrained=False)
        model = build_composite_model(backbone, {"label": task.head_spec})

        lit_module = LitModule(
            model=model,
            tasks=[task],
            optimizer_builder=OptimizerBuilder(base_lr=1e-3),
        )

        trainer = L.Trainer(
            max_epochs=3,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[
                CriterionScheduleCallback(task="label", parameter="gamma", start=2.0, end=0.5),
            ],
        )
        trainer.fit(lit_module, LitDataModule(plain_dm))

        focal_loss = task.criterion._loss  # noqa: SLF001 — pinning the applied end value
        assert isinstance(focal_loss, FocalLoss)
        assert focal_loss.gamma == pytest.approx(0.5)
        assert "schedule/label/gamma" in trainer.logged_metrics
