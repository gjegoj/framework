"""Export verification: numerical parity computation and the parity report."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.config import load_config
from src.export.entities import ExportReport, ParityResult
from src.export.pipeline import export_model
from src.export.verify import compute_parity
from src.models.assembly import build_composite_model
from src.models.backbones import TimmBackbone
from src.tasks import classification
from src.training.modules import LitModule
from src.training.optim import OptimizerBuilder
from tests.support.builders import raw_config as _raw


def _single_task_lit_module() -> LitModule:
    task = classification("label", num_classes=3)
    backbone = TimmBackbone("resnet18", pretrained=False)
    model = build_composite_model(backbone, {"label": task.head_spec})
    lit = LitModule(model=model, tasks=[task], optimizer_builder=OptimizerBuilder(base_lr=1e-3))
    lit.eval()
    return lit


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


class TestExportRaisesOnFailure:
    def test_export_raises_when_verification_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("onnx")
        failing = ExportReport(parity=ParityResult(9.9, 9.9, within_tolerance=False, per_output={}))
        monkeypatch.setattr("src.export.pipeline.verify_artifact", lambda *args, **kwargs: failing)
        lit = _single_task_lit_module()
        config = load_config(_raw(image_size=[32, 32], export={"targets": [{"format": "onnx"}], "combined": True}))
        with pytest.raises(RuntimeError, match="verification failed"):
            export_model(lit.model, lit.tasks, config, tmp_path)
