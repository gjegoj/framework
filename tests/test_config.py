"""Unit tests for the experiment config schema and loader."""

from typing import Any

import pytest

from src.config import ConfigError, ExperimentConfig, load_config


def _raw(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid raw config, with optional top-level overrides."""
    base: dict[str, Any] = {
        "project": "demo",
        "epochs": 5,
        "batch_size": 8,
        "lr": 1e-3,
        "image_size": [224, 224],
        "data": {
            "sources": "data/classification.csv",
            "inputs": "image_path",
            "split": {"train": 0.8, "val": 0.1, "test": 0.1},
        },
        "backbone": {"name": "resnet18"},
        "optimizer": {"lr": 1e-3},
        "tasks": {
            "label": {"preset": "classification", "target": "label", "class_mapping": {0: "cat", 1: "cow", 2: "dog"}}
        },
    }
    base.update(overrides)
    return base


class TestValidConfig:
    def test_minimal_config_builds(self) -> None:
        cfg = load_config(_raw())
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.project == "demo"
        assert cfg.image_size == (224, 224)  # list coerced to tuple
        assert cfg.seed == 42  # default
        assert cfg.mean == [0.485, 0.456, 0.406]  # ImageNet default
        assert cfg.backbone.kind == "timm"  # default

    def test_num_classes_derived_from_class_mapping(self) -> None:
        cfg = load_config(_raw())
        assert cfg.tasks["label"].class_mapping == {0: "cat", 1: "cow", 2: "dog"}
        assert cfg.tasks["label"].num_classes is None  # derived at runtime from class_mapping size

    def test_per_head_optimizer_override(self) -> None:
        raw = _raw()
        raw["tasks"]["label"]["optimizer"] = {"lr": 1e-4}
        cfg = load_config(raw)
        assert cfg.tasks["label"].optimizer is not None
        assert cfg.tasks["label"].optimizer.lr == 1e-4

    def test_task_extra_keys_preserved(self) -> None:
        raw = _raw()
        raw["tasks"]["label"]["head"] = {"_target_": "my.Head", "hidden": 256}
        cfg = load_config(raw)
        head = cfg.tasks["label"].head
        assert isinstance(head, dict)
        assert head["hidden"] == 256
        assert head["_target_"] == "my.Head"

    def test_loss_and_metrics_specs_parse(self) -> None:
        raw = _raw()
        raw["tasks"]["label"]["loss"] = {
            "name": "cross_entropy",
            "label_smoothing": 0.1,
        }
        raw["tasks"]["label"]["metrics"] = {"accuracy": {"top_k": 1}, "f1": None}
        cfg = load_config(raw)
        task = cfg.tasks["label"]
        assert task.loss == {"name": "cross_entropy", "label_smoothing": 0.1}
        assert task.metrics is not None
        assert task.metrics["accuracy"] == {"top_k": 1}
        assert task.metrics["f1"] is None

    def test_optimizer_extra_kwargs_preserved(self) -> None:
        raw = _raw()
        raw["optimizer"] = {"name": "sgd", "lr": 1e-2, "momentum": 0.9}
        cfg = load_config(raw)
        extra = cfg.optimizer.model_extra
        assert extra is not None and extra["momentum"] == 0.9


class TestInvalidConfig:
    def test_split_must_sum_to_one(self) -> None:
        with pytest.raises(ConfigError, match="sum to 1.0"):
            load_config(
                _raw(
                    data={
                        "sources": "d.csv",
                        "inputs": "p",
                        "split": {"train": 0.5, "val": 0.1, "test": 0.1},
                    }
                )
            )

    def test_split_rejects_predict(self) -> None:
        with pytest.raises(ConfigError, match="train/val/test"):
            load_config(
                _raw(
                    data={
                        "sources": "d.csv",
                        "inputs": "p",
                        "split": {"train": 0.5, "predict": 0.5},
                    }
                )
            )

    def test_image_size_must_be_positive(self) -> None:
        with pytest.raises(ConfigError, match="must be positive"):
            load_config(_raw(image_size=[0, 224]))

    def test_tasks_cannot_be_empty(self) -> None:
        with pytest.raises(ConfigError):
            load_config(_raw(tasks={}))

    def test_unknown_top_level_key_rejected(self) -> None:
        with pytest.raises(ConfigError):
            load_config(_raw(unexpected="oops"))

    def test_non_positive_lr_rejected(self) -> None:
        with pytest.raises(ConfigError):
            load_config(_raw(optimizer={"lr": 0.0}))
