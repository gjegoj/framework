"""Experiment assemblers: resolution, capability guards, detection assembly."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.composition.wiring import experiment_assemblers, resolve_experiment_assembler
from src.composition.wiring.detection import DetectionExperimentAssembler
from src.composition.wiring.experiment import StandardExperimentAssembler
from src.config import load_config
from src.config.schema import ExperimentConfig
from src.core.runtime import RuntimeContext
from src.data.detection import DetectionDataModule
from src.training import CompleteModelLitModule
from tests.support.builders import make_yolo_dataset
from tests.support.builders import minimal_config as _minimal_config

DETECTION_MODEL_SECTION = {"kind": "yolo", "name": "yolov8n.yaml"}


def _detection_config(tmp_path: Path, **overrides: object) -> ExperimentConfig:
    data_yaml = make_yolo_dataset(tmp_path)
    base: dict[str, object] = {
        "data": {"sources": str(data_yaml)},
        "model": dict(DETECTION_MODEL_SECTION),
        "tasks": {"boxes": {"preset": "detection"}},
        "image_size": [64, 64],
        "run_export": False,
    }
    base.update(overrides)
    return load_config(_minimal_config(**base))


class TestInputsGuardForBindingsContour:
    def test_standard_run_without_inputs_fails_loudly(self) -> None:
        """inputs became optional for detection; the bindings contour must still demand it."""
        from src.composition.wiring import build_bindings, build_data_module

        raw = _minimal_config()
        del raw["data"]["inputs"]
        config = load_config(raw)
        with pytest.raises(ValueError, match="data.inputs is required"):
            build_data_module(config, build_bindings(config), RuntimeContext())


class TestResolution:
    def test_yolo_kind_resolves_to_detection_assembler(self, tmp_path: Path) -> None:
        assert isinstance(resolve_experiment_assembler(_detection_config(tmp_path)), DetectionExperimentAssembler)

    def test_standard_config_resolves_to_standard_assembler(self) -> None:
        assert isinstance(resolve_experiment_assembler(load_config(_minimal_config())), StandardExperimentAssembler)

    def test_both_assemblers_registered(self) -> None:
        assert "standard" in experiment_assemblers and "detection" in experiment_assemblers


class TestKindDispatchCoherence:
    def test_yolo_kind_without_detection_preset_rejected(self, tmp_path: Path) -> None:
        config = _detection_config(
            tmp_path, tasks={"label": {"preset": "classification", "target": "label", "num_classes": 3}}
        )
        with pytest.raises(ValueError, match="preset"):
            resolve_experiment_assembler(config)

    def test_detection_preset_without_complete_model_kind_rejected(self, tmp_path: Path) -> None:
        data_yaml = make_yolo_dataset(tmp_path)
        raw = _minimal_config(
            data={"sources": str(data_yaml)}, tasks={"boxes": {"preset": "detection"}}, run_export=False
        )
        with pytest.raises(ValueError, match="kind"):
            resolve_experiment_assembler(load_config(raw))

    def test_backbone_extras_become_hyperparameters(self, tmp_path: Path) -> None:
        from typing import Any, cast

        config = _detection_config(tmp_path, model={**DETECTION_MODEL_SECTION, "box": 3.0})
        lit_module, _datamodule, _tasks = resolve_experiment_assembler(config).build(config, RuntimeContext())
        assert float(cast("Any", lit_module.model)._detection_model.args.box) == 3.0


class TestDetectionCapabilityGuards:
    def test_export_rejected(self, tmp_path: Path) -> None:
        config = _detection_config(tmp_path, run_export=True)
        with pytest.raises(ValueError, match="(?i)detection export"):
            resolve_experiment_assembler(config).validate(config)

    def test_lora_rejected(self, tmp_path: Path) -> None:
        config = _detection_config(tmp_path, lora={"target_modules": ["conv1"]})
        with pytest.raises(ValueError, match="LoRA"):
            resolve_experiment_assembler(config).validate(config)

    def test_distillation_rejected(self, tmp_path: Path) -> None:
        config = _detection_config(
            tmp_path, distillation={"teachers": [{"model": {"name": "resnet18"}, "ckpt_path": "t.ckpt"}]}
        )
        with pytest.raises(ValueError, match="distillation"):
            resolve_experiment_assembler(config).validate(config)

    def test_task_mixing_rejected(self, tmp_path: Path) -> None:
        data_yaml = make_yolo_dataset(tmp_path)
        raw = _minimal_config(
            data={"sources": str(data_yaml)},
            model=dict(DETECTION_MODEL_SECTION),
            tasks={
                "boxes": {"preset": "detection"},
                "label": {"preset": "classification", "target": "label", "num_classes": 3},
            },
            run_export=False,
        )
        config = load_config(raw)
        with pytest.raises(ValueError, match="preset|single detection task"):
            DetectionExperimentAssembler().validate(config)

    def test_missing_name_rejected(self, tmp_path: Path) -> None:
        config = _detection_config(tmp_path, model={"kind": "yolo"})
        with pytest.raises(ValueError, match="name"):
            resolve_experiment_assembler(config).validate(config)

    def test_happy_path_passes(self, tmp_path: Path) -> None:
        config = _detection_config(tmp_path)
        resolve_experiment_assembler(config).validate(config)  # should not raise


class TestDetectionBuild:
    def test_builds_module_datamodule_and_runtime_fact(self, tmp_path: Path) -> None:
        config = _detection_config(tmp_path)
        runtime = RuntimeContext()
        lit_module, lit_data_module, tasks = resolve_experiment_assembler(config).build(config, runtime)
        assert isinstance(lit_module, CompleteModelLitModule)
        assert isinstance(lit_data_module, DetectionDataModule)
        assert tasks == []
        assert runtime.num_classes["boxes"] == 2  # data facts flow through RuntimeContext
