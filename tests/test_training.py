"""Unit and smoke tests for the training layer."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import albumentations as A
import cv2
import lightning as L
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch import Tensor

from src.core.entities import LossResult, Task
from src.core.enums import Stage
from src.core.ports import Criterion
from src.core.runtime import RuntimeContext
from src.data import (
    AlbumentationsTransform,
    CsvDataSource,
    DataLoaderOptions,
    DataModule,
    LabelEncoder,
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
def csv_path(make_image_csv: Callable[..., Path]) -> Path:
    """15 synthetic 32x32 images, 3 classes."""
    return make_image_csv(count=15, size=32, seed=1)


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

        opt = OptimizerBuilder(base_lr=1e-3, task_lr_overrides={"label": 1e-4}).build(model)
        assert len(opt.param_groups) == 2
        lrs = {g["name"]: g["lr"] for g in opt.param_groups}
        assert lrs["backbone"] == pytest.approx(1e-3)
        assert lrs["head/label"] == pytest.approx(1e-4)

    def test_no_param_overlap_between_groups(self) -> None:
        backbone = TimmBackbone("resnet18", pretrained=False)
        from src.core.entities import HeadSpec

        model = build_composite_model(backbone, {"label": HeadSpec(kind="linear", out_features=3)})

        opt = OptimizerBuilder(base_lr=1e-3, task_lr_overrides={"label": 1e-4}).build(model)
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

    def test_build_optimizer_builder_binds_per_task_lr(self) -> None:
        from src.composition.wiring import build_optimizer_builder
        from src.config.schema import OptimizerConfig
        from src.core.entities import HeadSpec
        from src.models import build_composite_model

        builder = build_optimizer_builder(OptimizerConfig(name="adamw", lr=1e-3), {"label": 1e-4})
        model = build_composite_model(
            TimmBackbone("resnet18", pretrained=False), {"label": HeadSpec(kind="linear", out_features=3)}
        )
        lrs = {group["name"]: group["lr"] for group in builder.build(model).param_groups}
        assert lrs["head/label"] == pytest.approx(1e-4)  # override bound through the builder
        assert lrs["backbone"] == pytest.approx(1e-3)


# ------------------------------------ criterion with learnable parameters (InfoNCE/SigLIP)


class _ScaledCrossEntropy(Criterion):
    """A criterion carrying one learnable scalar — stands in for InfoNCE/SigLIP's logit_scale."""

    def __init__(self) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.zeros(()))

    def forward(self, logits: Tensor, target: Tensor) -> LossResult:
        value = F.cross_entropy(logits * self.logit_scale.exp(), target.long())
        return LossResult(total=value, components={"scaled_ce": value})


class TestCriterionParametersAreTrainable:
    """A criterion with learnable parameters must be optimized, checkpointed, and moved to device.

    Metric-learning losses (InfoNCE/SigLIP) hold learnable ``logit_scale``/``bias``; because the
    optimizer is built from ``self.model`` and the criteria live only on the plain ``self.tasks``
    list, those parameters were silently frozen and absent from checkpoints.
    """

    @staticmethod
    def _task_with_parametric_criterion() -> tuple[Task, nn.Parameter]:
        criterion = _ScaledCrossEntropy()
        task = classification("label", num_classes=3)
        return dataclasses.replace(task, criterion=criterion), criterion.logit_scale

    def _lit(self, task: Task) -> LitModule:
        from src.models.backbones import EmbeddingBackbone

        model = build_composite_model(EmbeddingBackbone(embedding_dim=8), {"label": task.head_spec})
        return LitModule(model=model, tasks=[task], optimizer_builder=OptimizerBuilder(base_lr=1e-3))

    def test_criterion_parameter_is_in_the_optimizer(self) -> None:
        task, scale = self._task_with_parametric_criterion()
        optimizer = cast("torch.optim.Optimizer", self._lit(task).configure_optimizers())
        optimized_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
        assert id(scale) in optimized_ids

    def test_criterion_parameter_is_a_registered_submodule(self) -> None:
        task, scale = self._task_with_parametric_criterion()
        lit = self._lit(task)
        assert id(scale) in {id(param) for param in lit.parameters()}  # → device move + checkpoint

    def test_criterion_parameter_is_saved_in_the_state_dict(self) -> None:
        task, _ = self._task_with_parametric_criterion()
        lit = self._lit(task)
        assert any(key.endswith("logit_scale") for key in lit.state_dict())


# ------------------------------------------------ LitModule smoke


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


class TestStepOutputContract:
    """The step returns StepOutput fields via on_*_batch_end."""

    def _lit(self) -> LitModule:
        from src.models.assembly import build_composite_model
        from src.models.backbones import EmbeddingBackbone
        from src.tasks import classification

        task = classification("label", num_classes=3)
        model = build_composite_model(EmbeddingBackbone(embedding_dim=8), {"label": task.head_spec})
        return LitModule(model=model, tasks=[task], optimizer_builder=OptimizerBuilder(base_lr=1e-3))

    def _batch(self) -> Any:
        import torch

        from src.core.entities import Batch
        from src.core.keys import IMAGE

        return Batch(inputs={IMAGE: torch.randn(4, 8)}, targets={"label": torch.tensor([0, 1, 2, 0])})

    def test_shared_step_returns_loss_and_task_views(self) -> None:
        import torch

        from src.core.entities import TaskStepView, is_step_output

        result = self._lit()._shared_step(self._batch(), Stage.TRAIN)
        assert is_step_output(result)
        assert isinstance(result["loss"], torch.Tensor) and result["loss"].ndim == 0
        assert "output" not in result  # vestigial raw ModelOutput dropped — task_views covers viz
        view = result["task_views"]["label"]
        assert isinstance(view, TaskStepView)
        assert view.predictions.shape == (4, 3)
        assert view.metric_target.shape == (4,)

    def test_task_view_preds_are_detached(self) -> None:
        """predictions feed metrics + visualization only (never backprop), so they carry no graph."""
        result = self._lit()._shared_step(self._batch(), Stage.TRAIN)
        assert result["task_views"]["label"].predictions.requires_grad is False

    def test_validation_step_returns_task_views_for_callbacks(self) -> None:
        """validation_step must return outputs (was None) so on_validation_batch_end gets them."""
        result = self._lit().validation_step(self._batch(), 0)
        assert result is not None
        assert "label" in result["task_views"]


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


class TestSchedulerRegistry:
    def test_builtin_schedulers_resolve(self) -> None:
        import torch.optim.lr_scheduler as sched

        from src.training.optim import schedulers

        assert schedulers.get("cosine") is sched.CosineAnnealingLR
        assert schedulers.get("onecycle") is sched.OneCycleLR
        assert schedulers.get("plateau") is sched.ReduceLROnPlateau
        assert schedulers.get("step") is sched.StepLR


class TestSchedulerBuilder:
    @staticmethod
    def _optimizer() -> torch.optim.Optimizer:
        return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)

    def test_cosine_builds_epoch_config_without_monitor(self) -> None:
        from src.training.optim import SchedulerBuilder

        builder = SchedulerBuilder.from_name("cosine", interval="epoch", extra_kwargs={"T_max": 5})
        config = builder.build(self._optimizer(), trainer_facts={})
        assert isinstance(config["scheduler"], torch.optim.lr_scheduler.CosineAnnealingLR)
        assert config["interval"] == "epoch"
        assert "monitor" not in config

    def test_plateau_includes_monitor(self) -> None:
        from src.training.optim import SchedulerBuilder

        builder = SchedulerBuilder.from_name("plateau", monitor="loss/val", extra_kwargs={"mode": "min"})
        config = builder.build(self._optimizer(), trainer_facts={})
        assert config["monitor"] == "loss/val"

    def test_runtime_kwargs_routes_trainer_fact(self) -> None:
        from src.training.optim import SchedulerBuilder

        builder = SchedulerBuilder.from_name(
            "onecycle",
            interval="step",
            runtime_kwargs={"total_steps": "total_steps"},
            extra_kwargs={"max_lr": 0.1},
        )
        config = builder.build(
            self._optimizer(), trainer_facts={"total_steps": 100, "steps_per_epoch": 10, "epochs": 5}
        )
        scheduler = config["scheduler"]
        assert isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)
        assert scheduler.total_steps == 100  # routed from trainer_facts["total_steps"]

    @staticmethod
    def _two_group_optimizer(backbone_lr: float = 1e-3, head_lr: float = 1e-4) -> torch.optim.Optimizer:
        """An optimizer with a per-head param group, mirroring OptimizerBuilder's output."""
        return torch.optim.SGD(
            [
                {"name": "backbone", "params": [torch.nn.Parameter(torch.zeros(1))], "lr": backbone_lr},
                {"name": "head/species", "params": [torch.nn.Parameter(torch.zeros(1))], "lr": head_lr},
            ]
        )

    def _build_onecycle(self, optimizer: torch.optim.Optimizer, **extra: Any) -> torch.optim.lr_scheduler.OneCycleLR:
        from src.training.optim import SchedulerBuilder

        builder = SchedulerBuilder.from_name(
            "onecycle",
            interval="step",
            runtime_kwargs={"total_steps": "total_steps"},
            extra_kwargs={"max_lr": 1e-3, **extra},
        )
        scheduler = builder.build(optimizer, trainer_facts={"total_steps": 100})["scheduler"]
        return cast(torch.optim.lr_scheduler.OneCycleLR, scheduler)

    def test_onecycle_scales_scalar_max_lr_per_head_group(self) -> None:
        """A scalar max_lr is expanded per group, scaled by each group's lr (per-head survives)."""
        optimizer = self._two_group_optimizer(backbone_lr=1e-3, head_lr=1e-4)
        self._build_onecycle(optimizer)  # max_lr=${lr}=1e-3 scalar
        peaks = [group["max_lr"] for group in optimizer.param_groups]
        assert peaks[0] == pytest.approx(1e-3)  # backbone keeps the configured peak
        assert peaks[1] == pytest.approx(1e-4)  # head/species scaled to its own lr

    def test_onecycle_explicit_list_max_lr_is_untouched(self) -> None:
        """An explicit per-group list wins — the builder does not override the user's choice."""
        optimizer = self._two_group_optimizer()
        self._build_onecycle(optimizer, max_lr=[5e-3, 5e-4])
        peaks = [group["max_lr"] for group in optimizer.param_groups]
        assert peaks == pytest.approx([5e-3, 5e-4])

    def test_onecycle_single_group_keeps_scalar(self) -> None:
        """No per-head overrides (one group) → scalar max_lr is honoured as-is."""
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
        self._build_onecycle(optimizer)
        assert optimizer.param_groups[0]["max_lr"] == pytest.approx(1e-3)

    def test_cosine_is_unaffected_by_group_scaling(self) -> None:
        """Schedulers without per-group LR params (cosine) keep each group's own base lr."""
        from src.training.optim import SchedulerBuilder

        optimizer = self._two_group_optimizer(backbone_lr=1e-3, head_lr=1e-4)
        builder = SchedulerBuilder.from_name("cosine", interval="epoch", extra_kwargs={"T_max": 5})
        builder.build(optimizer, trainer_facts={})
        assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([1e-3, 1e-4])


class TestSchedulerWiring:
    def test_build_scheduler_builder_none(self) -> None:
        from src.composition.wiring.training import build_scheduler_builder

        assert build_scheduler_builder(None) is None

    def test_build_scheduler_builder_rejects_unknown_fact(self) -> None:
        from src.composition.wiring.training import build_scheduler_builder
        from src.config.schema import SchedulerConfig

        cfg = SchedulerConfig.model_validate(
            {"name": "onecycle", "interval": "step", "runtime_kwargs": {"total_steps": "bogus"}}
        )
        with pytest.raises(ValueError, match="trainer fact"):
            build_scheduler_builder(cfg)

    def test_configure_optimizers_returns_scheduler_dict(self) -> None:
        from types import SimpleNamespace

        from src.training.optim import SchedulerBuilder

        task = classification("label", num_classes=3)
        model = build_composite_model(TimmBackbone("resnet18", pretrained=False), {"label": task.head_spec})
        builder = SchedulerBuilder.from_name(
            "onecycle",
            interval="step",
            runtime_kwargs={"total_steps": "total_steps"},
            extra_kwargs={"max_lr": 0.1},
        )
        lit = LitModule(
            model=model,
            tasks=[task],
            optimizer_builder=OptimizerBuilder(base_lr=1e-3),
            scheduler_builder=builder,
        )
        lit._trainer = SimpleNamespace(estimated_stepping_batches=50, num_training_batches=10, max_epochs=5)  # type: ignore[assignment]
        result = lit.configure_optimizers()
        assert isinstance(result, dict)
        lr_scheduler = cast(dict[str, Any], result)["lr_scheduler"]
        assert isinstance(lr_scheduler, dict)
        assert lr_scheduler["interval"] == "step"
        assert lr_scheduler["scheduler"].total_steps == 50

    def test_configure_optimizers_without_scheduler_returns_optimizer(self) -> None:
        task = classification("label", num_classes=3)
        model = build_composite_model(TimmBackbone("resnet18", pretrained=False), {"label": task.head_spec})
        lit = LitModule(model=model, tasks=[task], optimizer_builder=OptimizerBuilder(base_lr=1e-3))
        assert isinstance(lit.configure_optimizers(), torch.optim.Optimizer)


class TestTestVerbose:
    """``trainer.test`` should print Lightning's own table only when our bar is absent."""

    def test_suppressed_when_metrics_progress_bar_present(self) -> None:
        from types import SimpleNamespace

        from src.callbacks.progress_bar import MetricsProgressBar
        from src.composition.wiring.training import _lightning_prints_test_results

        trainer: Any = SimpleNamespace(callbacks=[MetricsProgressBar()])
        assert _lightning_prints_test_results(trainer) is False

    def test_kept_without_metrics_progress_bar(self) -> None:
        from types import SimpleNamespace

        from src.composition.wiring.training import _lightning_prints_test_results

        trainer: Any = SimpleNamespace(callbacks=[object()])
        assert _lightning_prints_test_results(trainer) is True
