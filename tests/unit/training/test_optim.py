"""Optimizer and LR-scheduler builders: param groups, registry lookup, wiring seams."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from src.models import build_composite_model
from src.models.backbones import TimmBackbone
from src.tasks import classification
from src.training import (
    LitModule,
    OptimizerBuilder,
)


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
