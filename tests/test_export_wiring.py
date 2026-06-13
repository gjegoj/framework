"""Unit tests for export wiring (checkpoint resolution, output dir, weight loading)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.composition.wiring.checkpointing import resolve_ckpt_file
from src.composition.wiring.export import ensure_module_weights_for_export, run_export
from src.config import load_config
from src.models.assembly import build_composite_model
from src.models.backbones import EmbeddingBackbone
from src.tasks import classification
from src.training.module import LitModule
from src.training.optimizer import OptimizerBuilder
from tests.test_config import _raw


def _lit_module() -> LitModule:
    task = classification("label", num_classes=3)
    model = build_composite_model(EmbeddingBackbone(embedding_dim=8), {"label": task.head_spec})
    return LitModule(model=model, tasks=[task], optimizer_builder=OptimizerBuilder(base_lr=1e-3))


class TestResolveCkptFile:
    def test_best_alias(self) -> None:
        trainer = MagicMock()
        trainer.checkpoint_callback = MagicMock(best_model_path="/tmp/best.ckpt")
        assert resolve_ckpt_file(trainer, "best") == "/tmp/best.ckpt"

    def test_last_alias(self) -> None:
        trainer = MagicMock()
        trainer.checkpoint_callback = MagicMock(last_model_path="/tmp/last.ckpt")
        assert resolve_ckpt_file(trainer, "last") == "/tmp/last.ckpt"

    def test_explicit_path_passthrough(self) -> None:
        trainer = MagicMock()
        assert resolve_ckpt_file(trainer, "/explicit.ckpt") == "/explicit.ckpt"

    def test_best_without_callback_raises(self) -> None:
        trainer = MagicMock()
        trainer.checkpoint_callback = None
        with pytest.raises(ValueError, match="best"):
            resolve_ckpt_file(trainer, "best")


class TestEnsureModuleWeightsForExport:
    def test_skips_when_tested(self) -> None:
        trainer = MagicMock()
        lit_module = MagicMock()
        config = load_config(_raw())
        with patch("src.composition.wiring.export.load_init_weights") as mock_load:
            ensure_module_weights_for_export(trainer, lit_module, config, trained=True, tested=True)
        mock_load.assert_not_called()

    def test_loads_weights_when_not_tested(self, tmp_path: Path) -> None:
        source = _lit_module()
        weight_key = "model.heads.label.fc.weight"
        original = source.state_dict()[weight_key].detach().clone()
        ckpt_path = tmp_path / "weights.ckpt"
        torch.save({"state_dict": source.state_dict()}, ckpt_path)

        target = _lit_module()
        trainer = MagicMock()
        trainer.checkpoint_callback = None
        config = load_config(_raw(ckpt_path=str(ckpt_path), run_train=False))

        ensure_module_weights_for_export(trainer, target, config, trained=False, tested=False)

        assert torch.allclose(target.state_dict()[weight_key], original)


class TestRunExport:
    def test_resolves_save_dir_export_subfolder(self, tmp_path: Path) -> None:
        config = load_config(_raw(save_dir=str(tmp_path), run_export=True, export={"formats": []}))
        trainer = MagicMock()
        lit_module = MagicMock()

        with patch("src.composition.wiring.export.export_model") as mock_export:
            run_export(trainer, lit_module, [], config, trained=True, tested=True)

        mock_export.assert_not_called()

    def test_calls_export_model_with_composite_model(self, tmp_path: Path) -> None:
        from src.models.backbones import TimmBackbone

        task = classification("label", num_classes=3)
        model = build_composite_model(TimmBackbone("resnet18", pretrained=False), {"label": task.head_spec})
        lit = LitModule(model=model, tasks=[task], optimizer_builder=OptimizerBuilder(base_lr=1e-3))
        config = load_config(
            _raw(
                save_dir=str(tmp_path),
                run_export=True,
                export={"formats": ["onnx"], "combined": True},
                image_size=[32, 32],
            )
        )
        trainer = MagicMock()

        with patch("src.composition.wiring.export.export_model") as mock_export:
            run_export(trainer, lit, lit.tasks, config, trained=True, tested=True)
        mock_export.assert_called_once_with(lit.model, lit.tasks, config, tmp_path / "export")
