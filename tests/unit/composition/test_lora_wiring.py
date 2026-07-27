"""LoRA wiring: conditional injection, the Freeze x LoRA guard, checkpoint-mode threading."""

from __future__ import annotations

import pytest

from src.composition.wiring import apply_lora_if_configured, build_lit_module, build_tasks, validate_lora_preconditions
from src.config import load_config
from src.config.schema import ExperimentConfig
from src.core.entities import Task
from src.core.runtime import RuntimeContext
from src.models import build_composite_model, has_lora_layers
from src.models.assembly import CompositeModel
from src.models.registry import backbones
from src.training import OptimizerBuilder
from tests.support.builders import minimal_config as _minimal_config

LORA_SECTION = {"target_modules": ["conv1"], "rank": 2}


def _config(**overrides: object) -> ExperimentConfig:
    return load_config(_minimal_config(**overrides))


def _model_and_tasks(config: ExperimentConfig) -> tuple[CompositeModel, list[Task]]:
    tasks = build_tasks(config, RuntimeContext(num_classes={"label": 3}))
    backbone = backbones.create("timm", name="resnet18", pretrained=False)
    model = build_composite_model(backbone, {task.name: task.head_spec for task in tasks})
    return model, tasks


class TestApplyLoraIfConfigured:
    def test_no_op_without_lora_section(self) -> None:
        config = _config()
        model, _ = _model_and_tasks(config)
        apply_lora_if_configured(config, model)
        assert not has_lora_layers(model)

    def test_injects_into_backbone_and_leaves_heads_trainable(self) -> None:
        config = _config(lora=LORA_SECTION)
        model, _ = _model_and_tasks(config)
        apply_lora_if_configured(config, model)
        assert has_lora_layers(model.backbone)
        assert all(parameter.requires_grad for parameter in model.heads.parameters())
        backbone_trainable = [name for name, p in model.backbone.named_parameters() if p.requires_grad]
        assert backbone_trainable and all("lora_" in name for name in backbone_trainable)


class TestValidateLoraPreconditions:
    def test_freeze_inside_backbone_rejected(self) -> None:
        config = _config(lora=LORA_SECTION, callbacks={"freeze": {"targets": ["model.backbone._encoder"]}})
        with pytest.raises(ValueError, match="LoRA owns backbone freezing"):
            validate_lora_preconditions(config)

    def test_freeze_outside_backbone_allowed(self) -> None:
        config = _config(lora=LORA_SECTION, callbacks={"freeze": {"targets": ["model.heads.label"]}})
        validate_lora_preconditions(config)  # should not raise

    def test_freeze_without_lora_allowed(self) -> None:
        config = _config(callbacks={"freeze": {"targets": ["model.backbone"]}})
        validate_lora_preconditions(config)  # should not raise


class TestExportMerge:
    def test_run_export_merges_adapters(self, tmp_path: object) -> None:
        """run_export folds LoRA into base weights before building artifacts."""
        import torch

        from src.composition.wiring.export import run_export

        config = _config(
            lora=LORA_SECTION,
            export={"targets": [{"format": "torchscript", "method": "trace"}], "output_dir": str(tmp_path)},
        )
        model, tasks = _model_and_tasks(config)
        apply_lora_if_configured(config, model)
        lit_module = build_lit_module(config, model, tasks, OptimizerBuilder(base_lr=1e-3))
        features = torch.randn(2, 3, 64, 64)
        lit_module.eval()
        with torch.no_grad():
            before = lit_module.model({"image": features}).task_logits["label"]

        from unittest.mock import MagicMock

        run_export(MagicMock(), lit_module, tasks, config, trained=False, tested=True)

        assert not has_lora_layers(lit_module.model)
        with torch.no_grad():
            after = lit_module.model({"image": features}).task_logits["label"]
        assert torch.allclose(before, after, atol=1e-5)


class TestCheckpointModeThreading:
    def test_lit_module_prunes_only_when_lora_configured(self) -> None:
        config = _config(lora=LORA_SECTION)
        model, tasks = _model_and_tasks(config)
        lit_module = build_lit_module(config, model, tasks, OptimizerBuilder(base_lr=1e-3))
        assert lit_module._checkpoint_trainable_only is True
        assert lit_module.strict_loading is False

    def test_lit_module_full_checkpoints_without_lora(self) -> None:
        config = _config()
        model, tasks = _model_and_tasks(config)
        lit_module = build_lit_module(config, model, tasks, OptimizerBuilder(base_lr=1e-3))
        assert lit_module._checkpoint_trainable_only is False
        assert lit_module.strict_loading is True
