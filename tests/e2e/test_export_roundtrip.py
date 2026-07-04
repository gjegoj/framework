"""Export round-trips through real backends: ONNX/TorchScript files, verification, TensorRT, embedder."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from src.config import load_config
from src.core.keys import ENCODER_LAST
from src.export.entities import ExportRequest
from src.export.onnx import OnnxExporter
from src.export.pipeline import export_model
from src.export.registry import exporters
from src.export.wrapper import CombinedExportModel
from src.models.assembly import build_composite_model
from src.models.backbones import SmpBackbone, TimmBackbone
from src.tasks import classification, segmentation
from src.training.modules import LitModule
from src.training.optim import OptimizerBuilder
from tests.support.builders import raw_config as _raw


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


class TestExportOnnx:
    def test_combined_onnx_export(self, tmp_path: Path) -> None:
        pytest.importorskip("onnx")
        import onnx

        lit = _single_task_lit_module()
        config = load_config(_raw(image_size=[32, 32], export={"targets": [{"format": "onnx"}], "combined": True}))
        artifacts = export_model(lit.model, lit.tasks, config, tmp_path)
        assert len(artifacts) == 1
        assert artifacts[0].path.suffix == ".onnx"
        model = onnx.load(str(artifacts[0].path))
        onnx.checker.check_model(model)
        assert [node.name for node in model.graph.input] == ["input"]
        assert [node.name for node in model.graph.output] == ["output"]

    def test_static_batch_onnx_export(self, tmp_path: Path) -> None:
        """dynamic_batch=False skips the dry-run forward (no dynamic axes); the graph must
        still export valid with a *static* batch dimension (dynamic_axes=None path)."""
        pytest.importorskip("onnx")
        import onnx

        lit = _single_task_lit_module()
        config = load_config(
            _raw(
                image_size=[32, 32], export={"targets": [{"format": "onnx", "dynamic_batch": False}], "combined": True}
            )
        )
        artifacts = export_model(lit.model, lit.tasks, config, tmp_path)
        model = onnx.load(str(artifacts[0].path))
        onnx.checker.check_model(model)
        batch_dim = model.graph.input[0].type.tensor_type.shape.dim[0]
        assert batch_dim.dim_param == "", f"expected static batch, got dim_param={batch_dim.dim_param!r}"
        assert batch_dim.dim_value == 1

    def test_simplify_produces_valid_onnx(self, tmp_path: Path) -> None:
        pytest.importorskip("onnx")
        pytest.importorskip("onnxsim")
        import onnx

        lit = _single_task_lit_module()
        config = load_config(
            _raw(
                image_size=[32, 32],
                export={"targets": [{"format": "onnx", "simplify": True}], "combined": True},
            )
        )
        artifacts = export_model(lit.model, lit.tasks, config, tmp_path)
        model = onnx.load(str(artifacts[0].path))
        onnx.checker.check_model(model)
        # dynamic batch must survive simplification
        assert model.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "batch"

    def test_combined_multitask_generic_io_names(self, tmp_path: Path) -> None:
        pytest.importorskip("onnx")
        import onnx

        lit = _multitask_lit_module()
        config = load_config(
            _raw(
                image_size=[32, 32],
                export={"targets": [{"format": "onnx"}], "combined": True, "split_components": False},
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
                    "targets": [{"format": "onnx"}],
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
                export={"targets": [{"format": "onnx"}], "combined": True, "split_components": False},
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
                export={"targets": [{"format": "onnx"}], "combined": False, "split_components": True},
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
                export={"targets": [{"format": "onnx"}], "combined": True, "split_components": True},
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
        config = load_config(
            _raw(image_size=[32, 32], export={"targets": [{"format": "torchscript"}], "combined": True})
        )
        artifacts = export_model(lit.model, lit.tasks, config, tmp_path)
        assert len(artifacts) == 1
        assert artifacts[0].path.suffix == ".pt"
        traced = torch.jit.load(str(artifacts[0].path))
        outputs = traced(torch.randn(1, 3, 32, 32))
        probs = outputs[0] if isinstance(outputs, tuple) else outputs
        assert probs.shape == (1, 3)

    def test_script_method_compiles_a_scriptable_module(self, tmp_path: Path) -> None:
        from src.export.entities import ExportRequest

        class _Double(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2

        path = tmp_path / "double.pt"
        request = ExportRequest(
            module=_Double(),
            example_inputs=(torch.zeros(2),),
            path=path,
            input_names=["input"],
            output_names=["output"],
            options={"method": "script"},
        )
        exporters.create("torchscript").export(request)
        loaded = torch.jit.load(str(path))
        assert torch.allclose(loaded(torch.ones(2)), torch.full((2,), 2.0))

    def test_script_method_raises_readable_error_for_composite_wrapper(self, tmp_path: Path) -> None:
        lit = _single_task_lit_module()
        config = load_config(
            _raw(
                image_size=[32, 32],
                export={"targets": [{"format": "torchscript", "method": "script"}], "combined": True},
            )
        )
        with pytest.raises(RuntimeError, match="method='trace'"):
            export_model(lit.model, lit.tasks, config, tmp_path)


class TestOnnxVerification:
    def test_combined_onnx_export_reports_parity_and_checks(self, tmp_path: Path) -> None:
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
        lit = _single_task_lit_module()
        config = load_config(
            _raw(
                image_size=[32, 32],
                export={"targets": [{"format": "onnx", "infer_shapes": True}], "combined": True},
            )
        )
        artifacts = export_model(lit.model, lit.tasks, config, tmp_path)
        report = artifacts[0].report
        assert report is not None
        assert report.parity is not None and report.parity.within_tolerance
        assert set(report.checks) == {"onnx.checker", "shape_inference"}
        assert report.ok is True

    def test_verify_outputs_false_skips_parity(self, tmp_path: Path) -> None:
        pytest.importorskip("onnx")
        lit = _single_task_lit_module()
        config = load_config(
            _raw(
                image_size=[32, 32],
                export={"targets": [{"format": "onnx", "verify_outputs": False}], "combined": True},
            )
        )
        artifacts = export_model(lit.model, lit.tasks, config, tmp_path)
        assert artifacts[0].report is not None
        assert artifacts[0].report.parity is None  # parity skipped, checker still ran


class TestTorchScriptVerification:
    def test_torchscript_export_reports_parity_no_static_checks(self, tmp_path: Path) -> None:
        lit = _single_task_lit_module()
        config = load_config(
            _raw(image_size=[32, 32], export={"targets": [{"format": "torchscript"}], "combined": True})
        )
        artifacts = export_model(lit.model, lit.tasks, config, tmp_path)
        report = artifacts[0].report
        assert report is not None
        assert report.checks == {}  # TorchScript has no static validators
        assert report.parity is not None and report.parity.within_tolerance


class TestTensorRtExporter:
    def test_export_without_cuda_raises(self, tmp_path: Path) -> None:
        from src.export.entities import ExportRequest
        from src.export.tensorrt import TensorRtExporter

        if torch.cuda.is_available():
            pytest.skip("CUDA present — the no-CUDA guard does not trigger here.")
        request = ExportRequest(
            module=torch.nn.Identity(),
            example_inputs=(torch.randn(1, 3, 8, 8),),
            path=tmp_path / "model.plan",
            input_names=["input"],
            output_names=["output"],
            options={"precision": "fp16"},
        )
        with pytest.raises(RuntimeError, match="requires a CUDA device"):
            TensorRtExporter().export(request)

    def test_profile_shapes_explicit_and_fallback(self) -> None:
        from src.export.tensorrt import _profile_shapes

        example = torch.randn(1, 3, 8, 8)
        # Fallback: batch 1/4/8 over the example's own C, H, W.
        assert _profile_shapes({}, example) == ([1, 3, 8, 8], [4, 3, 8, 8], [8, 3, 8, 8])
        # Explicit profile wins.
        shapes = {"min": [1, 3, 16, 16], "opt": [2, 3, 16, 16], "max": [4, 3, 16, 16]}
        assert _profile_shapes({"shapes": shapes}, example) == ([1, 3, 16, 16], [2, 3, 16, 16], [4, 3, 16, 16])

    @pytest.mark.skipif(
        not torch.cuda.is_available() or importlib.util.find_spec("torch_tensorrt") is None,
        reason="TensorRT round-trip needs CUDA + torch-tensorrt (GPU node only).",
    )
    def test_engine_round_trip(self, tmp_path: Path) -> None:
        from src.export.entities import ExportRequest
        from src.export.tensorrt import TensorRtExporter

        module = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(8, 3, 3, padding=1)
        ).eval()
        request = ExportRequest(
            module=module,
            example_inputs=(torch.randn(1, 3, 16, 16),),
            path=tmp_path / "model_combined.plan",
            input_names=["input"],
            output_names=["output"],
            options={"precision": "fp32", "min_block_size": 1},
        )
        exporter = TensorRtExporter()
        exporter.export(request)
        assert request.path.exists()
        assert next(module.parameters()).device.type == "cpu"  # restored after export

        runner = exporter.load(request.path)
        out = runner({"input": torch.randn(1, 3, 16, 16)})
        assert len(out) == 1 and out[0].shape == (1, 3, 16, 16)


class TestEmbedderExport:
    def test_exported_embedder_outputs_unit_norm_and_no_prototypes(self, tmp_path: Path) -> None:
        pytest.importorskip("onnx")
        import onnx

        from src.models.backbones import EmbeddingBackbone
        from src.tasks.presets import task_presets

        # EmbeddingBackbone (no image encoder) is the cheapest exportable backbone here;
        # export_model's plan always builds an image-shaped dummy input, so this test drives
        # the same CombinedExportModel + OnnxExporter machinery export_model uses internally,
        # directly — mirroring TestExportOnnx's construction, not a new export entry point.
        task = task_presets.create("arcface_embedding")("embed", num_classes=16, class_count=5)
        model = build_composite_model(EmbeddingBackbone(embedding_dim=8), {"embed": task.head_spec})
        model.eval()
        wrapper = CombinedExportModel(model, ("embed",), {"embed": task.activation})
        example = torch.randn(2, 8)
        outputs = wrapper(example)
        embedding = outputs[0] if isinstance(outputs, tuple) else outputs
        assert embedding.shape == (2, 16)
        assert torch.allclose(embedding.norm(dim=-1), torch.ones(2), atol=1e-5)  # NormalizeActivation applied

        onnx_path = tmp_path / "model_combined.onnx"
        request = ExportRequest(
            module=wrapper,
            example_inputs=(example,),
            path=onnx_path,
            input_names=["input"],
            output_names=["output"],
        )
        OnnxExporter().export(request)

        initializer_shapes = {tuple(tensor.dims) for tensor in onnx.load(str(onnx_path)).graph.initializer}
        assert (16, 5) not in initializer_shapes and (5, 16) not in initializer_shapes  # no prototypes in graph
