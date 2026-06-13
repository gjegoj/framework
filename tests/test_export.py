"""Unit tests for model export (wrappers, ONNX pipeline, config)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.config import load_config
from src.core.keys import ENCODER_LAST, POOLED
from src.export.onnx import OnnxExporter
from src.export.pipeline import export_model, resolve_export_io_names
from src.export.registry import exporters
from src.export.spec import build_export_plan
from src.export.wrapper import BackboneExportModel, CombinedExportModel, HeadExportModel
from src.models.assembly import build_composite_model
from src.models.backbones import SmpBackbone, TimmBackbone
from src.models.heads import LinearHead
from src.tasks import classification, segmentation, triplet
from src.training.module import LitModule
from src.training.optimizer import OptimizerBuilder
from tests.test_config import _raw


def _images(batch: int, size: int = 32) -> torch.Tensor:
    return torch.randn(batch, 3, size, size)


def _multitask_lit_module() -> LitModule:
    species = classification("species", num_classes=5, feature_key=ENCODER_LAST)
    mask = segmentation("mask", num_classes=3)
    backbone = SmpBackbone(encoder_name="resnet18", pretrained=False)
    model = build_composite_model(
        backbone,
        {"species": species.head_spec, "mask": mask.head_spec},
    )
    lit = LitModule(
        model=model,
        tasks=[species, mask],
        optimizer_builder=OptimizerBuilder(base_lr=1e-3),
    )
    lit.eval()
    return lit


def _single_task_lit_module() -> LitModule:
    task = classification("label", num_classes=3)
    backbone = TimmBackbone("resnet18", pretrained=False)
    model = build_composite_model(backbone, {"label": task.head_spec})
    lit = LitModule(model=model, tasks=[task], optimizer_builder=OptimizerBuilder(base_lr=1e-3))
    lit.eval()
    return lit


class TestExportWrappers:
    def test_combined_wrapper_output_count(self) -> None:
        lit = _multitask_lit_module()
        activations = {task.name: task.activation for task in lit.tasks}
        wrapper = CombinedExportModel(lit.model, ("species", "mask"), activations)
        outputs = wrapper(_images(2))
        assert isinstance(outputs, tuple)
        assert len(outputs) == 2
        assert outputs[0].shape == (2, 5)
        assert outputs[1].shape[:2] == (2, 3)

    def test_combined_wrapper_applies_task_activation(self) -> None:
        lit = _single_task_lit_module()
        task = lit.tasks[0]
        wrapper = CombinedExportModel(lit.model, ("label",), {"label": task.activation})
        probs = wrapper(_images(2))[0]
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_backbone_wrapper_streams(self) -> None:
        backbone = TimmBackbone("resnet18", pretrained=False)
        backbone.eval()
        wrapper = BackboneExportModel(backbone, (POOLED,))
        outputs = wrapper(_images(1))
        assert len(outputs) == 1
        assert outputs[0].ndim == 2

    def test_head_wrapper_applies_activation(self) -> None:
        from src.tasks.activations import SoftmaxActivation

        head = LinearHead(8, 3)
        wrapper = HeadExportModel(head, activation=SoftmaxActivation())
        probs = wrapper(torch.randn(2, 8))
        assert probs.shape == (2, 3)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)


class TestExportConfig:
    def test_invalid_format_raises(self) -> None:
        with pytest.raises(Exception, match="unknown format"):
            load_config(_raw(export={"formats": ["coreml"]}))

    def test_tensorrt_format_rejected(self) -> None:
        with pytest.raises(Exception, match="unknown format"):
            load_config(_raw(export={"formats": ["tensorrt"]}))

    def test_empty_formats_allowed(self) -> None:
        config = load_config(_raw(export={"formats": []}))
        assert config.export.formats == []

    def test_all_formats_preset(self) -> None:
        config = load_config(_raw(export={"formats": ["onnx", "torchscript"]}))
        assert config.export.formats == ["onnx", "torchscript"]

    def test_generic_io_names_default(self) -> None:
        config = load_config(_raw())
        assert config.export.generic_io_names is True


class TestExportIoNames:
    def test_single_tensor_uses_bare_prefix(self) -> None:
        assert resolve_export_io_names(["image"], prefix="input", generic=True) == ["input"]
        assert resolve_export_io_names(["label"], prefix="output", generic=True) == ["output"]

    def test_multiple_tensors_use_index_suffix(self) -> None:
        assert resolve_export_io_names(["a", "b"], prefix="output", generic=True) == ["output_0", "output_1"]

    def test_semantic_names_when_disabled(self) -> None:
        names = ["species", "mask"]
        assert resolve_export_io_names(names, prefix="output", generic=False) == names


class TestExporterPort:
    def test_registered_extensions(self) -> None:
        assert isinstance(exporters.create("onnx"), OnnxExporter)
        assert exporters.create("onnx").extension == ".onnx"
        assert exporters.create("torchscript").extension == ".pt"


class TestExportOnnx:
    def test_combined_onnx_export(self, tmp_path: Path) -> None:
        pytest.importorskip("onnx")
        import onnx

        lit = _single_task_lit_module()
        config = load_config(_raw(image_size=[32, 32], export={"formats": ["onnx"], "combined": True}))
        artifacts = export_model(lit.model, lit.tasks, config, tmp_path)
        assert len(artifacts) == 1
        assert artifacts[0].path.suffix == ".onnx"
        model = onnx.load(str(artifacts[0].path))
        onnx.checker.check_model(model)
        assert [node.name for node in model.graph.input] == ["input"]
        assert [node.name for node in model.graph.output] == ["output"]

    def test_combined_multitask_generic_io_names(self, tmp_path: Path) -> None:
        pytest.importorskip("onnx")
        import onnx

        lit = _multitask_lit_module()
        config = load_config(
            _raw(
                image_size=[32, 32],
                export={"formats": ["onnx"], "combined": True, "split_components": False},
            )
        )
        artifacts = export_model(lit.model, lit.tasks, config, tmp_path)
        model = onnx.load(str(artifacts[0].path))
        assert [node.name for node in model.graph.input] == ["input"]
        assert [node.name for node in model.graph.output] == ["output_0", "output_1"]

    def test_semantic_io_names_when_disabled(self, tmp_path: Path) -> None:
        pytest.importorskip("onnx")
        import onnx

        lit = _multitask_lit_module()
        config = load_config(
            _raw(
                image_size=[32, 32],
                export={
                    "formats": ["onnx"],
                    "combined": True,
                    "split_components": False,
                    "generic_io_names": False,
                },
            )
        )
        artifacts = export_model(lit.model, lit.tasks, config, tmp_path)
        model = onnx.load(str(artifacts[0].path))
        assert [node.name for node in model.graph.input] == ["image"]
        assert [node.name for node in model.graph.output] == ["species", "mask"]

    def test_combined_multitask_outputs(self, tmp_path: Path) -> None:
        pytest.importorskip("onnx")
        import onnx

        lit = _multitask_lit_module()
        config = load_config(
            _raw(
                image_size=[32, 32],
                export={"formats": ["onnx"], "combined": True, "split_components": False},
            )
        )
        artifacts = export_model(lit.model, lit.tasks, config, tmp_path)
        model = onnx.load(str(artifacts[0].path))
        assert len(model.graph.output) == 2

    def test_split_components_writes_three_files(self, tmp_path: Path) -> None:
        pytest.importorskip("onnx")
        lit = _multitask_lit_module()
        config = load_config(
            _raw(
                image_size=[32, 32],
                export={"formats": ["onnx"], "combined": False, "split_components": True},
            )
        )
        artifacts = export_model(lit.model, lit.tasks, config, tmp_path)
        kinds = {artifact.kind for artifact in artifacts}
        assert kinds == {"backbone", "head"}
        assert sum(1 for artifact in artifacts if artifact.kind == "head") == 2
        assert (tmp_path / "backbone.onnx").is_file()

    def test_combined_and_split_together(self, tmp_path: Path) -> None:
        pytest.importorskip("onnx")
        lit = _multitask_lit_module()
        config = load_config(
            _raw(
                image_size=[32, 32],
                export={"formats": ["onnx"], "combined": True, "split_components": True},
            )
        )
        artifacts = export_model(lit.model, lit.tasks, config, tmp_path)
        assert (tmp_path / "model_combined.onnx").is_file()
        assert (tmp_path / "backbone.onnx").is_file()
        assert (tmp_path / "head_species.onnx").is_file()
        assert (tmp_path / "head_mask.onnx").is_file()
        assert len(artifacts) == 4


class TestExportTorchScript:
    def test_combined_torchscript_export(self, tmp_path: Path) -> None:
        lit = _single_task_lit_module()
        config = load_config(_raw(image_size=[32, 32], export={"formats": ["torchscript"], "combined": True}))
        artifacts = export_model(lit.model, lit.tasks, config, tmp_path)
        assert len(artifacts) == 1
        assert artifacts[0].path.suffix == ".pt"
        traced = torch.jit.load(str(artifacts[0].path))
        outputs = traced(torch.randn(1, 3, 32, 32))
        probs = outputs[0] if isinstance(outputs, tuple) else outputs
        assert probs.shape == (1, 3)


class TestExportGuards:
    def test_ranking_topology_rejected(self) -> None:
        task = triplet("rank", num_classes=16)
        backbone = TimmBackbone("resnet18", pretrained=False)
        model = build_composite_model(backbone, {"rank": task.head_spec})
        config = load_config(_raw(image_size=[32, 32]))
        with pytest.raises(ValueError, match="ranking"):
            build_export_plan(model, [task], config)
