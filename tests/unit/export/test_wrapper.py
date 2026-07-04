"""Export wrappers and the exporter port: combined/per-part graphs, io names, guards."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.composition.wiring.export import validate_export_preconditions
from src.config import load_config
from src.core.keys import ENCODER_LAST, POOLED
from src.export.entities import ExportRequest
from src.export.onnx import OnnxExporter
from src.export.pipeline import resolve_export_io_names
from src.export.registry import exporters
from src.export.spec import build_export_plan
from src.export.wrapper import BackboneExportModel, CombinedExportModel, HeadExportModel
from src.models.assembly import build_composite_model
from src.models.backbones import SmpBackbone, TimmBackbone
from src.models.heads import LinearHead
from src.tasks import classification, segmentation, triplet
from src.training.modules import LitModule
from src.training.optim import OptimizerBuilder
from tests.support.builders import raw_config as _raw


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


class TestExportIoNames:
    @pytest.mark.parametrize(
        ("names", "prefix", "generic", "expected"),
        [
            pytest.param(["image"], "input", True, ["input"], id="single-input-bare-prefix"),
            pytest.param(["label"], "output", True, ["output"], id="single-output-bare-prefix"),
            pytest.param(["a", "b"], "output", True, ["output_0", "output_1"], id="multi-indexed"),
            pytest.param(["species", "mask"], "output", False, ["species", "mask"], id="semantic-when-disabled"),
        ],
    )
    def test_resolves_io_names(self, names: list[str], prefix: str, generic: bool, expected: list[str]) -> None:
        assert resolve_export_io_names(names, prefix=prefix, generic=generic) == expected


class TestExporterPort:
    def test_registered_extensions(self) -> None:
        assert isinstance(exporters.create("onnx"), OnnxExporter)
        assert exporters.create("onnx").extension == ".onnx"
        assert exporters.create("torchscript").extension == ".pt"
        assert exporters.create("tensorrt").extension == ".plan"


class TestExporterPortDefaults:
    def test_default_load_and_validate_are_noops(self) -> None:
        from src.export.entities import ExportRequest
        from src.export.ports import ModelExporter

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


class TestExportGuards:
    def test_multiview_topology_rejected(self) -> None:
        task = triplet("rank", num_classes=16)
        backbone = TimmBackbone("resnet18", pretrained=False)
        model = build_composite_model(backbone, {"rank": task.head_spec})
        config = load_config(_raw(image_size=[32, 32]))
        with pytest.raises(ValueError, match="multiview"):
            build_export_plan(model, [task], config)

    def test_preconditions_reject_multiview_when_export_enabled(self) -> None:
        task = triplet("rank", num_classes=16)
        config = load_config(_raw(image_size=[32, 32], run_export=True))
        with pytest.raises(ValueError, match="multiview"):
            validate_export_preconditions(config, [task])

    def test_preconditions_noop_when_export_disabled(self) -> None:
        task = triplet("rank", num_classes=16)
        config = load_config(_raw(image_size=[32, 32], run_train=True, run_export=False))
        validate_export_preconditions(config, [task])  # must not raise

    def test_preconditions_pass_for_exportable_topology(self) -> None:
        task = classification("label", num_classes=3)
        config = load_config(_raw(image_size=[32, 32], run_export=True))
        validate_export_preconditions(config, [task])  # must not raise
