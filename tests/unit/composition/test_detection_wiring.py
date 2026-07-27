"""Detection wiring: run detection, fail-fast guards, experiment assembly."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.composition.wiring import build_detection_experiment, is_detection_run, validate_detection_preconditions
from src.config import load_config
from src.config.schema import ExperimentConfig
from src.data.detection import DetectionDataModule
from src.training import DetectionLitModule
from tests.support.builders import make_yolo_dataset
from tests.support.builders import minimal_config as _minimal_config

DETECTION_TASK = {"preset": "detection", "model": "yolov8n.yaml"}


def _detection_config(tmp_path: Path, **overrides: object) -> ExperimentConfig:
    data_yaml = make_yolo_dataset(tmp_path)
    base: dict[str, object] = {
        "data": {"sources": str(data_yaml)},
        "tasks": {"boxes": dict(DETECTION_TASK)},
        "image_size": [64, 64],
        "run_export": False,
    }
    base.update(overrides)
    return load_config(_minimal_config(**base))


class TestInputsGuardForBindingsContour:
    def test_standard_run_without_inputs_fails_loudly(self) -> None:
        """inputs became optional for detection; the bindings contour must still demand it."""
        from src.composition.wiring import build_bindings, build_data_module
        from src.core.runtime import RuntimeContext

        raw = _minimal_config()
        del raw["data"]["inputs"]
        config = load_config(raw)
        with pytest.raises(ValueError, match="data.inputs is required"):
            build_data_module(config, build_bindings(config), RuntimeContext())


class TestIsDetectionRun:
    def test_true_for_detection_preset(self, tmp_path: Path) -> None:
        assert is_detection_run(_detection_config(tmp_path))

    def test_false_for_standard_tasks(self) -> None:
        assert not is_detection_run(load_config(_minimal_config()))


class TestValidateDetectionPreconditions:
    def test_mixing_with_other_tasks_rejected(self, tmp_path: Path) -> None:
        data_yaml = make_yolo_dataset(tmp_path)
        raw = _minimal_config(
            data={"sources": str(data_yaml)},
            tasks={
                "boxes": dict(DETECTION_TASK),
                "label": {"preset": "classification", "target": "label", "num_classes": 3},
            },
            run_export=False,
        )
        with pytest.raises(ValueError, match="single detection task"):
            validate_detection_preconditions(load_config(raw))

    def test_export_rejected(self, tmp_path: Path) -> None:
        config = _detection_config(tmp_path, run_export=True)
        with pytest.raises(ValueError, match="(?i)detection export"):
            validate_detection_preconditions(config)

    def test_lora_rejected(self, tmp_path: Path) -> None:
        config = _detection_config(tmp_path, lora={"target_modules": ["conv1"]})
        with pytest.raises(ValueError, match="LoRA"):
            validate_detection_preconditions(config)

    def test_distillation_rejected(self, tmp_path: Path) -> None:
        config = _detection_config(
            tmp_path,
            distillation={"teachers": [{"backbone": {"name": "resnet18"}, "ckpt_path": "t.ckpt"}]},
        )
        with pytest.raises(ValueError, match="distillation"):
            validate_detection_preconditions(config)

    def test_missing_model_rejected(self, tmp_path: Path) -> None:
        data_yaml = make_yolo_dataset(tmp_path)
        raw = _minimal_config(
            data={"sources": str(data_yaml)},
            tasks={"boxes": {"preset": "detection"}},
            run_export=False,
        )
        with pytest.raises(ValueError, match="model"):
            validate_detection_preconditions(load_config(raw))

    def test_happy_path_passes(self, tmp_path: Path) -> None:
        validate_detection_preconditions(_detection_config(tmp_path))  # should not raise


class TestBuildDetectionExperiment:
    def test_builds_module_and_datamodule(self, tmp_path: Path) -> None:
        lit_module, datamodule = build_detection_experiment(_detection_config(tmp_path))
        assert isinstance(lit_module, DetectionLitModule)
        assert isinstance(datamodule, DetectionDataModule)
        assert datamodule.num_classes == 2
