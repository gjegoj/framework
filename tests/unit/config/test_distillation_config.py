"""DistillationConfig validation: teachers required, positive temperature, non-negative weights."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.schema import DistillationConfig

TEACHER = {"backbone": {"kind": "timm", "name": "resnet18"}, "ckpt_path": "runs/teacher.ckpt"}


class TestDistillationConfig:
    def test_minimal_valid(self) -> None:
        config = DistillationConfig(teachers=[TEACHER])
        assert config.temperature == 2.0
        assert config.weight == 1.0
        assert config.loss is None and config.tasks is None
        assert config.teachers[0].backbone.name == "resnet18"

    def test_empty_teachers_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DistillationConfig(teachers=[])

    @pytest.mark.parametrize("temperature", [0.0, -1.0])
    def test_non_positive_temperature_rejected(self, temperature: float) -> None:
        with pytest.raises(ValidationError):
            DistillationConfig(teachers=[TEACHER], temperature=temperature)

    @pytest.mark.parametrize("weight", [-0.1, {"mask": -1.0}], ids=["float", "per_task"])
    def test_negative_weight_rejected(self, weight: object) -> None:
        with pytest.raises(ValidationError):
            DistillationConfig(teachers=[TEACHER], weight=weight)

    def test_per_task_weight_accepted(self) -> None:
        assert DistillationConfig(teachers=[TEACHER], weight={"mask": 0.7}).weight == {"mask": 0.7}

    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param({"teachers": [TEACHER], "heat": 8.0}, id="unknown_section_key"),
            pytest.param({"teachers": [{**TEACHER, "weights_file": "x.ckpt"}]}, id="unknown_teacher_key"),
        ],
    )
    def test_unknown_keys_rejected(self, raw: dict[str, object]) -> None:
        """extra='forbid': an unknown key (e.g. a typo'd field) must fail loudly, not silently keep the default.

        The probe keys are real words, not misspellings, on purpose: the `typos` pre-commit
        hook auto-corrects literal misspellings (e.g. ``temperature`` with a dropped last
        letter), which would silently turn this test's invalid input into valid config.
        """
        with pytest.raises(ValidationError):
            DistillationConfig(**raw)

    def test_experiment_config_defaults_to_disabled(self) -> None:
        from src.config.schema import ExperimentConfig

        assert ExperimentConfig.model_fields["distillation"].default is None
