"""CPU training smokes: classification, embedding, ArcFace embedder, segmentation, criterion schedule, detection."""

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
from src.callbacks.ema import EmaCallback
from src.callbacks.ema_checkpoint import EmaModelCheckpoint
from src.callbacks.freeze import FreezeCallback
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

        focal_loss = task.criterion._loss
        assert isinstance(focal_loss, FocalLoss)
        assert focal_loss.gamma == pytest.approx(0.5)
        assert "schedule/label/gamma" in trainer.logged_metrics


class TestCallbackComboSmoke:
    """freeze + EMA + criterion_schedule + EMA-aware checkpoint composing in one fit.

    Each callback is pinned in isolation by its unit tests; this smoke pins the
    cross-callback invariants a real Lightning fit exercises together: the schedule's
    cached criterion reference survives EMA's weight swaps, the frozen backbone stays
    untouched by training (EMA's final copy-in reproduces constants only up to float
    rounding), and the weights-only checkpoint holds the EMA weights that produced the
    monitored validation metric.
    """

    def test_freeze_ema_schedule_and_checkpoint_compose(self, csv_path: Path, tmp_path: Path) -> None:
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
        frozen_before = next(lit_module.model.backbone.parameters()).detach().clone()
        # monitor=None -> the last epoch is saved, so the file's EMA weights match the
        # post-fit module (on_train_end copies the average in; no updates follow the save).
        checkpoint = EmaModelCheckpoint(dirpath=tmp_path, filename="combo", save_weights_only=True)

        trainer = L.Trainer(
            max_epochs=3,
            accelerator="cpu",
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            callbacks=[
                FreezeCallback(targets=["model.backbone"]),
                CriterionScheduleCallback(task="label", parameter="gamma", start=2.0, end=0.5),
                EmaCallback(decay=0.9, warmup_fraction=0.0, use_buffers=True),
                checkpoint,
            ],
        )
        trainer.fit(lit_module, LitDataModule(plain_dm))

        # Schedule applied through EMA's swaps: the cached criterion reference stayed live.
        focal_loss = task.criterion._loss
        assert isinstance(focal_loss, FocalLoss)
        assert focal_loss.gamma == pytest.approx(0.5)

        # Backbone frozen: untouched by training, reproduced by the EMA copy-in up to rounding.
        frozen_after = next(lit_module.model.backbone.parameters()).detach()
        assert not frozen_after.requires_grad
        assert torch.allclose(frozen_after, frozen_before, atol=1e-6)

        # The weights-only checkpoint holds the EMA weights (== post-fit module weights).
        saved = torch.load(checkpoint.best_model_path, weights_only=False)
        for key, value in lit_module.state_dict().items():
            if value.dtype.is_floating_point:
                assert torch.equal(saved["state_dict"][key], value), f"checkpoint diverges at {key}"


class TestDistillationSmoke:
    """2-epoch CPU fit distilling a student from one frozen teacher.

    Pins the regime end-to-end: the KL term is logged, all losses stay finite, the
    teacher never trains, and the teacher weights never enter the student's state_dict.
    """

    def test_distillation_fit(self, csv_path: Path) -> None:
        from src.losses.distillation import KLDivergenceCriterion
        from src.models import TeacherEnsemble
        from src.training import DistillationLitModule

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
        head_specs = {"label": task.head_spec}
        student = build_composite_model(TimmBackbone("resnet18", pretrained=False), head_specs)
        teacher = build_composite_model(TimmBackbone("resnet18", pretrained=False), head_specs)
        teacher_weight_before = next(teacher.parameters()).detach().clone()
        ensemble = TeacherEnsemble([teacher])

        lit_module = DistillationLitModule(
            model=student,
            tasks=[task],
            optimizer_builder=OptimizerBuilder(base_lr=1e-3),
            teachers=ensemble,
            distillation_criteria={"label": KLDivergenceCriterion(temperature=2.0)},
            distillation_weights={"label": 0.7},
        )

        trainer = L.Trainer(
            max_epochs=2,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(lit_module, LitDataModule(plain_dm))

        assert "loss/train/label/kl" in trainer.logged_metrics
        assert all(torch.isfinite(value).all() for value in trainer.logged_metrics.values())
        # Teacher never trained and never leaked into the student's state_dict.
        assert torch.equal(next(teacher.parameters()).detach(), teacher_weight_before)
        assert not any("teacher" in key.lower() for key in lit_module.state_dict())


class TestLoraSmoke:
    """2-epoch CPU LoRA fit: frozen base, pruned checkpoint, resume, merge parity.

    resnet18 with ``conv1`` targets exercises the Conv2d adapter path end to end.
    """

    def test_lora_fit_checkpoint_and_merge(self, csv_path: Path, tmp_path: Path) -> None:
        from src.callbacks.ema_checkpoint import EmaModelCheckpoint
        from src.config.schema import LoraConfig
        from src.models import apply_lora, has_lora_layers, merge_lora

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
        backbone = TimmBackbone("resnet18", pretrained=False)
        model = build_composite_model(backbone, {"label": task.head_spec})
        apply_lora(model.backbone, LoraConfig(target_modules=["conv1"], rank=2))

        lit_module = LitModule(
            model=model,
            tasks=[task],
            optimizer_builder=OptimizerBuilder(base_lr=1e-3),
            checkpoint_trainable_only=True,
        )
        checkpoint = EmaModelCheckpoint(dirpath=tmp_path, filename="lora", save_weights_only=True)
        trainer = L.Trainer(
            max_epochs=2,
            accelerator="cpu",
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            callbacks=[checkpoint],
        )
        trainer.fit(lit_module, LitDataModule(plain_dm))

        # Pruned checkpoint: adapters + heads present, frozen base PARAMETERS absent,
        # buffers (BatchNorm running stats) kept by design.
        saved = torch.load(checkpoint.best_model_path, weights_only=False)
        saved_keys = set(saved["state_dict"].keys())
        assert any("lora_" in key for key in saved_keys)
        assert any(key.startswith("model.heads.") for key in saved_keys)
        frozen_parameter_keys = {
            f"model.{name}" for name, parameter in lit_module.model.named_parameters() if not parameter.requires_grad
        }
        assert frozen_parameter_keys and not (frozen_parameter_keys & saved_keys)
        buffer_keys = {f"model.{name}" for name, _ in lit_module.model.named_buffers()}
        assert buffer_keys & saved_keys

        # Resume path: a freshly built identical module loads the pruned checkpoint.
        fresh_model = build_composite_model(TimmBackbone("resnet18", pretrained=False), {"label": task.head_spec})
        apply_lora(fresh_model.backbone, LoraConfig(target_modules=["conv1"], rank=2))
        fresh_module = LitModule(
            model=fresh_model,
            tasks=[task],
            optimizer_builder=OptimizerBuilder(base_lr=1e-3),
            checkpoint_trainable_only=True,
        )
        fresh_module.load_state_dict(saved["state_dict"], strict=False)
        trained_adapters = {k: v for k, v in lit_module.state_dict().items() if "lora_" in k}
        for key, value in trained_adapters.items():
            assert torch.equal(dict(fresh_module.state_dict())[key], value), key

        # Merge parity on the trained model.
        lit_module.eval()
        features = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            before = lit_module.model({"image": features}).task_logits["label"]
        merge_lora(lit_module.model)
        with torch.no_grad():
            after = lit_module.model({"image": features}).task_logits["label"]
        assert not has_lora_layers(lit_module.model)
        assert torch.allclose(before, after, atol=1e-5)


class TestDetectionSmoke:
    """2-epoch CPU YOLO detection fit through the wiring entry point.

    Pins the regime end-to-end: all logged losses stay finite, mAP is computed and
    logged on val, and the EMA weights-only checkpoint loads back into a freshly
    built identical module.
    """

    def test_detection_fit_ema_checkpoint_roundtrip(self, tmp_path: Path) -> None:
        from src.composition.wiring import build_detection_experiment, validate_detection_preconditions
        from src.config import load_config
        from tests.support.builders import make_yolo_dataset, minimal_config

        data_yaml = make_yolo_dataset(tmp_path)
        config = load_config(
            minimal_config(
                data={"sources": str(data_yaml)},
                tasks={"boxes": {"preset": "detection", "model": "yolov8n.yaml"}},
                image_size=[64, 64],
                epochs=2,
                run_export=False,
            )
        )
        validate_detection_preconditions(config)
        lit_module, datamodule = build_detection_experiment(config)

        checkpoint = EmaModelCheckpoint(dirpath=tmp_path, filename="detect", save_weights_only=True)
        trainer = L.Trainer(
            max_epochs=2,
            accelerator="cpu",
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            callbacks=[EmaCallback(decay=0.9, warmup_fraction=0.0, use_buffers=True), checkpoint],
        )
        trainer.fit(lit_module, datamodule)

        assert "boxes/map50_95/val" in trainer.logged_metrics
        assert all(torch.isfinite(value).all() for value in trainer.logged_metrics.values())

        # The weights-only EMA checkpoint loads back into a freshly built identical module.
        saved = torch.load(checkpoint.best_model_path, weights_only=False)
        fresh_module, _ = build_detection_experiment(config)
        fresh_module.load_state_dict(saved["state_dict"])
