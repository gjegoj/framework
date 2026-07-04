"""Task-layer wiring: bindings, task assembly from config, and the arcface_embedding cell."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from src.composition.wiring import (
    build_bindings,
    build_tasks,
)
from src.config import load_config
from src.core.runtime import RuntimeContext
from src.data.encoders import LabelEncoder, MultiLabelEncoder, ScalarEncoder
from tests.support.builders import minimal_config as _minimal_config


class TestBuildTasks:
    def test_infers_num_classes_from_runtime(self) -> None:
        config = load_config(_minimal_config())
        runtime = RuntimeContext(num_classes={"label": 3})
        tasks = build_tasks(config, runtime)
        assert len(tasks) == 1
        assert tasks[0].name == "label"
        assert tasks[0].head_spec.out_features == 3

    def test_explicit_num_classes_takes_priority(self) -> None:
        raw = _minimal_config()
        raw["tasks"]["label"]["num_classes"] = 5
        config = load_config(raw)
        runtime = RuntimeContext(num_classes={"label": 3})  # would infer 3
        tasks = build_tasks(config, runtime)
        assert tasks[0].head_spec.out_features == 5

    def test_missing_num_classes_raises(self) -> None:
        config = load_config(_minimal_config())
        runtime = RuntimeContext()  # empty — no inferred classes
        with pytest.raises(ValueError, match="num_classes for task 'label'"):
            build_tasks(config, runtime)

    def test_regression_uses_dim(self) -> None:
        raw = _minimal_config()
        raw["tasks"] = {"score": {"preset": "regression", "target": "score", "dim": 1}}
        config = load_config(raw)
        runtime = RuntimeContext()  # dim bypasses num_classes lookup
        tasks = build_tasks(config, runtime)
        assert tasks[0].head_spec.out_features == 1


class TestBuildBindings:
    def test_classification_gets_label_encoder(self) -> None:
        config = load_config(_minimal_config())
        bindings = build_bindings(config)
        assert len(bindings) == 1
        assert isinstance(bindings[0].encoder, LabelEncoder)
        assert bindings[0].column == "label"

    def test_regression_gets_float_codec(self) -> None:
        raw = _minimal_config()
        raw["tasks"] = {"score": {"preset": "regression", "target": "score", "dim": 1}}
        config = load_config(raw)
        bindings = build_bindings(config)
        assert isinstance(bindings[0].encoder, ScalarEncoder)

    def test_targetless_task_gets_null_encoder_and_no_column(self) -> None:
        from src.data.encoders import NullTargetEncoder

        raw = _minimal_config()
        raw["tasks"] = {"rank": {"preset": "triplet"}}  # no 'target' → structure-only supervision
        config = load_config(raw)
        bindings = build_bindings(config)
        assert isinstance(bindings[0].encoder, NullTargetEncoder)
        assert bindings[0].column is None

    def test_multilabel_gets_binarize_codec(self) -> None:
        raw = _minimal_config()
        raw["tasks"] = {
            "tags": {
                "preset": "classification",
                "target": "tags",
                "objective": "multilabel",
            }
        }
        config = load_config(raw)
        bindings = build_bindings(config)
        assert isinstance(bindings[0].encoder, MultiLabelEncoder)

    def test_custom_separator_via_target_encoder_spec(self) -> None:
        raw = _minimal_config()
        raw["tasks"] = {
            "tags": {
                "preset": "classification",
                "target": "tags",
                "objective": "multilabel",
                "target_encoder": {"name": "multilabel", "separator": "|"},
            }
        }
        config = load_config(raw)
        bindings = build_bindings(config)
        assert isinstance(bindings[0].encoder, MultiLabelEncoder)
        assert bindings[0].encoder._separator == "|"

    def test_target_encoder_override(self) -> None:
        raw = _minimal_config()
        raw["tasks"]["label"]["target_encoder"] = "label"
        config = load_config(raw)
        bindings = build_bindings(config)
        assert isinstance(bindings[0].encoder, LabelEncoder)

    def test_segmentation_gets_mask_codec(self) -> None:
        from src.data.encoders import MaskEncoder

        raw = _minimal_config()
        raw["tasks"] = {"mask": {"preset": "segmentation", "target": "mask_path", "num_classes": 4}}
        config = load_config(raw)
        bindings = build_bindings(config)
        assert isinstance(bindings[0].encoder, MaskEncoder)


class TestArcFaceEmbeddingWiring:
    """Pins the arcface_embedding preset end-to-end: config dict → wired Task.

    Parts (encoder default, dimension seam, activation) landed in prior tasks and are
    unit-tested there; this integration test may already pass — that is expected, it is
    pinning the seam rather than driving new red/green cycles.
    """

    def test_embedder_task_wired_from_config(self, make_image_csv: Callable[..., Path]) -> None:
        from src.losses.angular import ProxyAngularCriterion
        from src.tasks.activations import NormalizeActivation

        csv_path = make_image_csv(count=15, size=32, seed=3)
        raw = _minimal_config()
        raw["data"]["sources"] = str(csv_path)
        raw["tasks"] = {"embed": {"preset": "arcface_embedding", "target": "label", "dim": 16}}
        config = load_config(raw)

        bindings = build_bindings(config)
        assert type(bindings[0].encoder).__name__ == "LabelEncoder"  # preset default_encoder="label"

        runtime = RuntimeContext()
        runtime.num_classes["embed"] = 3  # what DataModule.setup() would infer from the label column
        task = build_tasks(config, runtime)[0]
        assert task.head_spec.out_features == 16
        assert isinstance(task.activation, NormalizeActivation)
        assert isinstance(task.criterion, ProxyAngularCriterion)
        assert task.criterion.prototypes.shape == (16, 3)
