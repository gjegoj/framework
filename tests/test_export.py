"""Unit tests for model export (wrappers, ONNX pipeline, config)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.composition.wiring.export import validate_export_preconditions
from src.config import ConfigError, load_config
from src.config.export import ExportConfig
from src.core.keys import ENCODER_LAST, POOLED
from src.export.entities import ExportReport, ParityResult
from src.export.onnx import OnnxExporter
from src.export.pipeline import export_model, resolve_export_io_names
from src.export.registry import exporters
from src.export.spec import build_export_plan
from src.export.verify import compute_parity
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
        with pytest.raises(ConfigError):
            load_config(_raw(export={"targets": [{"format": "coreml"}]}))

    def test_tensorrt_format_rejected(self) -> None:
        with pytest.raises(ConfigError):
            load_config(_raw(export={"targets": [{"format": "tensorrt"}]}))

    def test_empty_targets_allowed(self) -> None:
        config = load_config(_raw(export={"targets": []}))
        assert config.export.targets == []

    def test_default_target_is_onnx(self) -> None:
        config = load_config(_raw())
        assert [t.format for t in config.export.targets] == ["onnx"]

    def test_onnx_and_torchscript_targets(self) -> None:
        config = load_config(_raw(export={"targets": [{"format": "onnx"}, {"format": "torchscript"}]}))
        assert [t.format for t in config.export.targets] == ["onnx", "torchscript"]

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
        config = load_config(_raw(image_size=[32, 32], export={"targets": [{"format": "onnx"}], "combined": True}))
        artifacts = export_model(lit.model, lit.tasks, config, tmp_path)
        assert len(artifacts) == 1
        assert artifacts[0].path.suffix == ".onnx"
        model = onnx.load(str(artifacts[0].path))
        onnx.checker.check_model(model)
        assert [node.name for node in model.graph.input] == ["input"]
        assert [node.name for node in model.graph.output] == ["output"]

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
        from src.core.entities import ExportRequest

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


class TestExportGuards:
    def test_ranking_topology_rejected(self) -> None:
        task = triplet("rank", num_classes=16)
        backbone = TimmBackbone("resnet18", pretrained=False)
        model = build_composite_model(backbone, {"rank": task.head_spec})
        config = load_config(_raw(image_size=[32, 32]))
        with pytest.raises(ValueError, match="ranking"):
            build_export_plan(model, [task], config)

    def test_preconditions_reject_ranking_when_export_enabled(self) -> None:
        task = triplet("rank", num_classes=16)
        config = load_config(_raw(image_size=[32, 32], run_export=True))
        with pytest.raises(ValueError, match="ranking"):
            validate_export_preconditions(config, [task])

    def test_preconditions_noop_when_export_disabled(self) -> None:
        task = triplet("rank", num_classes=16)
        config = load_config(_raw(image_size=[32, 32], run_train=True, run_export=False))
        validate_export_preconditions(config, [task])  # must not raise

    def test_preconditions_pass_for_exportable_topology(self) -> None:
        task = classification("label", num_classes=3)
        config = load_config(_raw(image_size=[32, 32], run_export=True))
        validate_export_preconditions(config, [task])  # must not raise


class TestExportYamlConfigs:
    def test_all_export_group_files_validate(self) -> None:
        import yaml

        group_dir = Path("configs/export")
        for path in sorted(group_dir.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text()) or {}
            if not raw:  # fully-commented placeholder (tensorrt.yaml)
                continue
            ExportConfig.model_validate(raw)  # must not raise


class TestExportReport:
    def test_ok_true_when_checks_pass_and_parity_within(self) -> None:
        report = ExportReport(
            checks={"onnx.checker": ""},
            parity=ParityResult(1e-6, 1e-6, within_tolerance=True, per_output={}),
        )
        assert report.ok is True

    def test_ok_false_when_a_check_fails(self) -> None:
        assert ExportReport(checks={"onnx.checker": "bad graph"}).ok is False

    def test_ok_false_when_parity_exceeds(self) -> None:
        assert ExportReport(parity=ParityResult(9.9, 9.9, within_tolerance=False, per_output={})).ok is False

    def test_ok_true_when_empty(self) -> None:
        assert ExportReport().ok is True

    def test_failure_summary_lists_failures(self) -> None:
        report = ExportReport(
            checks={"onnx.checker": "bad", "shape_inference": ""},
            parity=ParityResult(9.9, 1.0, within_tolerance=False, per_output={}),
        )
        summary = report.failure_summary
        assert "onnx.checker" in summary
        assert "shape_inference" not in summary  # passed check is not listed
        assert "parity" in summary

    def test_artifact_report_defaults_none(self) -> None:
        from src.export.entities import ExportArtifact

        assert ExportArtifact(path=Path("m.onnx"), format="onnx", kind="combined").report is None


class TestExporterPortDefaults:
    def test_default_load_and_validate_are_noops(self) -> None:
        from src.core.entities import ExportRequest
        from src.core.ports import ModelExporter

        class _Dummy(ModelExporter):
            @property
            def extension(self) -> str:
                return ".x"

            def export(self, request: ExportRequest) -> None:
                return None

        dummy = _Dummy()
        request = ExportRequest(
            module=torch.nn.Identity(),
            example_inputs=(torch.zeros(1),),
            path=Path("m.x"),
            input_names=["input"],
            output_names=["output"],
        )
        assert dummy.load(Path("m.x")) is None
        assert dummy.validate(request) == {}


class TestComputeParity:
    def test_identical_outputs_within_tolerance(self) -> None:
        ref = (torch.ones(2, 3),)
        result = compute_parity(ref, (ref[0].clone(),), ["output"], atol=1e-6, rtol=1e-6)
        assert result.within_tolerance is True
        assert result.max_abs_error == 0.0
        assert result.per_output == {"output": (0.0, 0.0)}

    def test_perturbed_outputs_exceed_tolerance(self) -> None:
        result = compute_parity((torch.zeros(2, 3),), (torch.full((2, 3), 0.5),), ["y"], atol=1e-4, rtol=1e-4)
        assert result.within_tolerance is False
        assert result.max_abs_error == pytest.approx(0.5)

    def test_per_output_keyed_by_name(self) -> None:
        ref = (torch.ones(1, 2), torch.ones(1, 4))
        result = compute_parity(ref, (ref[0].clone(), ref[1].clone()), ["a", "b"], atol=1e-6, rtol=1e-6)
        assert set(result.per_output) == {"a", "b"}


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


class TestExportRaisesOnFailure:
    def test_export_raises_when_verification_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("onnx")
        failing = ExportReport(parity=ParityResult(9.9, 9.9, within_tolerance=False, per_output={}))
        monkeypatch.setattr("src.export.pipeline.verify_artifact", lambda *args, **kwargs: failing)
        lit = _single_task_lit_module()
        config = load_config(_raw(image_size=[32, 32], export={"targets": [{"format": "onnx"}], "combined": True}))
        with pytest.raises(RuntimeError, match="verification failed"):
            export_model(lit.model, lit.tasks, config, tmp_path)
