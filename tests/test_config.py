"""Unit tests for the experiment config schema and loader."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.config import ConfigError, ExperimentConfig, load_config
from src.config.export import ExportConfig, OnnxOptions, TorchScriptOptions


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
        assert cfg.run_test is True  # default
        assert cfg.run_train is True  # default
        assert cfg.run_export is True  # default
        assert cfg.mean == [0.485, 0.456, 0.406]  # ImageNet default
        assert cfg.backbone.kind == "timm"  # default

    def test_run_test_can_be_disabled(self) -> None:
        cfg = load_config(_raw(run_test=False))
        assert cfg.run_test is False

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

    def test_unknown_data_key_rejected(self) -> None:
        """A typo in a data field (e.g. ``split_stratifi``) must fail loudly, not be swallowed."""
        with pytest.raises(ConfigError):
            load_config(
                _raw(
                    data={
                        "sources": "d.csv",
                        "inputs": "p",
                        "split": {"train": 0.8, "val": 0.1, "test": 0.1},
                        "split_stratifi": "label",  # typo of split_stratify
                    }
                )
            )

    def test_non_positive_lr_rejected(self) -> None:
        with pytest.raises(ConfigError):
            load_config(_raw(optimizer={"lr": 0.0}))


class TestExportConfigSchema:
    def test_default_target_is_onnx(self) -> None:
        cfg = ExportConfig()
        assert len(cfg.targets) == 1
        assert isinstance(cfg.targets[0], OnnxOptions)
        assert cfg.targets[0].opset_version == 17
        assert cfg.targets[0].dynamic_batch is True

    def test_onnx_options_round_trip(self) -> None:
        cfg = ExportConfig.model_validate({"targets": [{"format": "onnx", "opset_version": 13}]})
        assert isinstance(cfg.targets[0], OnnxOptions)
        assert cfg.targets[0].opset_version == 13

    def test_torchscript_target_parsed(self) -> None:
        cfg = ExportConfig.model_validate({"targets": [{"format": "torchscript"}]})
        assert isinstance(cfg.targets[0], TorchScriptOptions)

    def test_onnx_option_under_torchscript_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExportConfig.model_validate({"targets": [{"format": "torchscript", "opset_version": 17}]})

    def test_unknown_format_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExportConfig.model_validate({"targets": [{"format": "coreml"}]})

    def test_empty_targets_allowed(self) -> None:
        cfg = ExportConfig.model_validate({"targets": []})
        assert cfg.targets == []

    def test_onnx_simplify_default_false(self) -> None:
        assert OnnxOptions().simplify is False

    def test_simplify_under_torchscript_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExportConfig.model_validate({"targets": [{"format": "torchscript", "simplify": True}]})

    def test_torchscript_method_default_trace(self) -> None:
        assert TorchScriptOptions().method == "trace"

    def test_torchscript_method_script_accepted(self) -> None:
        cfg = ExportConfig.model_validate({"targets": [{"format": "torchscript", "method": "script"}]})
        target = cfg.targets[0]
        assert isinstance(target, TorchScriptOptions)
        assert target.method == "script"

    def test_torchscript_method_invalid_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExportConfig.model_validate({"targets": [{"format": "torchscript", "method": "bogus"}]})

    def test_verification_defaults_on_onnx(self) -> None:
        opt = OnnxOptions()
        assert opt.verify_outputs is True
        assert opt.atol == 1e-4
        assert opt.rtol == 1e-3
        assert opt.check_model is True
        assert opt.infer_shapes is False

    def test_torchscript_inherits_verification_fields(self) -> None:
        opt = TorchScriptOptions(atol=1e-2)
        assert opt.verify_outputs is True
        assert opt.atol == 1e-2

    def test_check_model_under_torchscript_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExportConfig.model_validate({"targets": [{"format": "torchscript", "check_model": True}]})

    def test_negative_atol_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExportConfig.model_validate({"targets": [{"format": "onnx", "atol": -1.0}]})


class TestSchedulerConfig:
    def test_scheduler_defaults_none(self) -> None:
        assert load_config(_raw()).scheduler is None

    def test_scheduler_parsed_with_extras_and_runtime_kwargs(self) -> None:
        cfg = load_config(
            _raw(
                scheduler={
                    "name": "onecycle",
                    "interval": "step",
                    "max_lr": 0.1,
                    "runtime_kwargs": {"total_steps": "total_steps"},
                }
            )
        )
        assert cfg.scheduler is not None
        assert cfg.scheduler.name == "onecycle"
        assert cfg.scheduler.runtime_kwargs == {"total_steps": "total_steps"}
        assert cfg.scheduler.model_dump()["max_lr"] == 0.1

    def test_scheduler_name_required(self) -> None:
        with pytest.raises(ConfigError):
            load_config(_raw(scheduler={"interval": "step"}))

    def test_scheduler_interval_validated(self) -> None:
        with pytest.raises(ConfigError):
            load_config(_raw(scheduler={"name": "cosine", "interval": "minute"}))


class TestSchedulerYamlConfigs:
    def test_active_scheduler_group_files_validate(self) -> None:
        import yaml

        from src.config.schema import SchedulerConfig

        for name in ("cosine", "onecycle", "plateau"):
            raw = yaml.safe_load(Path(f"configs/scheduler/{name}.yaml").read_text())
            for key, value in list(raw.items()):
                if isinstance(value, str) and "${" in value:
                    # Stub runtime interpolations: a bare ${...} stands in for a number
                    # (${lr}/${epochs}); a composite (${key:LOSS}/val/${key:TOTAL}) is a string.
                    whole_interpolation = value.startswith("${") and value.endswith("}") and value.count("${") == 1
                    raw[key] = 1 if whole_interpolation else "loss/val/total"
            SchedulerConfig.model_validate(raw)


class TestCacheConfig:
    def test_cache_defaults_none(self) -> None:
        assert load_config(_raw()).data.cache is None

    def test_cache_parsed(self) -> None:
        cfg = load_config(_raw(data={**_raw()["data"], "cache": {"ram_fraction": 0.5, "max_gb": 16}}))
        assert cfg.data.cache is not None
        assert cfg.data.cache.ram_fraction == 0.5
        assert cfg.data.cache.max_gb == 16
        assert cfg.data.cache.workers == 8  # default

    def test_cache_fraction_out_of_range_rejected(self) -> None:
        with pytest.raises(ConfigError):
            load_config(_raw(data={**_raw()["data"], "cache": {"ram_fraction": 1.5}}))

    def test_cache_unknown_key_rejected(self) -> None:
        with pytest.raises(ConfigError):
            load_config(_raw(data={**_raw()["data"], "cache": {"ram_fraction": 0.5, "bogus": 1}}))


class TestRotationExperiment:
    """The online-rotation experiment needs no custom encoder — the default label encoder
    yields the class index in ``load`` and the Rotate90WithLabel aug bumps it."""

    def test_rotation_task_validates_with_default_encoder(self) -> None:
        raw = _raw(
            tasks={
                "rotation": {
                    "preset": "classification",
                    "target": "rotation",
                    "class_mapping": {0: "0", 1: "90", 2: "180", 3: "270"},
                }
            }
        )
        task = load_config(raw).tasks["rotation"]
        assert task.preset == "classification"
        assert task.target == "rotation"
        assert task.class_mapping == {0: "0", 1: "90", 2: "180", 3: "270"}
        assert task.target_encoder is None  # default label encoder; the index lives in encoder.load

    def test_rotation_transforms_config_wires_the_aug(self) -> None:
        from omegaconf import OmegaConf

        from src.core.instantiate import instantiate
        from src.transforms import Rotate90WithLabel

        context = OmegaConf.create(
            {
                "seed": 0,
                "image_size": [128, 128],
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
                "transforms": OmegaConf.load("configs/transforms/rotation.yaml"),
            }
        )
        train_pipeline = OmegaConf.to_container(context.transforms.train, resolve=True)
        compose = instantiate(train_pipeline)
        assert any(isinstance(step, Rotate90WithLabel) for step in compose.transforms)
