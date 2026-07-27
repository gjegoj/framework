"""Detection fields on TaskConfig: model + hyperparameters accepted and optional."""

from __future__ import annotations

from src.config.schema import TaskConfig


class TestDetectionTaskConfig:
    def test_detection_fields_accepted(self) -> None:
        config = TaskConfig(preset="detection", model="yolo26n.yaml", hyperparameters={"mosaic": 1.0, "box": 7.5})
        assert config.model == "yolo26n.yaml"
        assert config.hyperparameters == {"mosaic": 1.0, "box": 7.5}

    def test_fields_default_to_none_for_other_presets(self) -> None:
        config = TaskConfig(preset="classification", target="label")
        assert config.model is None
        assert config.hyperparameters is None
