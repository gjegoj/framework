"""Unit tests for composition wiring (no Hydra required)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.callbacks.batch_transform import BatchTransformCallback
from src.composition.wiring import (
    build_bindings,
    build_data_source,
    build_lit_module,
    build_logger,
    build_task_lr_overrides,
    build_tasks,
    build_transforms,
)
from src.config import load_config
from src.core.enums import Stage
from src.core.runtime import RuntimeContext
from src.data.codecs import FloatCodec, LabelIndexCodec, MultiLabelBinarizeCodec
from src.data.sources import CsvDataSource, JsonDataSource

_RESIZE_NORMALIZE_TOTENSOR = [
    {"_target_": "albumentations.Resize", "height": 64, "width": 64},
    {"_target_": "albumentations.Normalize"},
    {"_target_": "albumentations.pytorch.ToTensorV2"},
]

_MINIMAL_TRANSFORMS: dict[str, Any] = {
    "train": {"_target_": "albumentations.Compose", "transforms": _RESIZE_NORMALIZE_TOTENSOR},
    "val": {"_target_": "albumentations.Compose", "transforms": _RESIZE_NORMALIZE_TOTENSOR},
}


def _minimal_config(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "project": "test",
        "epochs": 1,
        "batch_size": 4,
        "lr": 1e-3,
        "image_size": [64, 64],
        "data": {
            "sources": "data.csv",
            "inputs": "image_path",
            "split": {"train": 0.8, "val": 0.1, "test": 0.1},
        },
        "backbone": {"name": "resnet18"},
        "optimizer": {"lr": 1e-3},
        "tasks": {
            "label": {"preset": "classification", "target": "label", "class_mapping": {0: "cat", 1: "cow", 2: "dog"}}
        },
        "transforms": _MINIMAL_TRANSFORMS,
    }
    base.update(overrides)
    return base


class TestInstantiateNested:
    def test_builds_albumentations_resize(self) -> None:
        import albumentations as A

        from src.core.instantiate import instantiate

        spec = {"_target_": "albumentations.Resize", "height": 32, "width": 32}
        obj = instantiate(spec)
        assert isinstance(obj, A.Resize)

    def test_builds_compose_with_nested_transforms(self) -> None:
        import albumentations as A

        from src.core.instantiate import instantiate

        spec = {
            "_target_": "albumentations.Compose",
            "transforms": [
                {"_target_": "albumentations.HorizontalFlip", "p": 0.5},
                {"_target_": "albumentations.Normalize"},
            ],
        }
        obj = instantiate(spec)
        assert isinstance(obj, A.Compose)
        assert len(obj.transforms) == 2

    def test_drops_hydra_meta_keys(self) -> None:
        import albumentations as A

        from src.core.instantiate import instantiate

        spec = {
            "_target_": "albumentations.HorizontalFlip",
            "_convert_": "partial",
            "p": 0.3,
        }
        obj = instantiate(spec)
        assert isinstance(obj, A.HorizontalFlip)

    def test_plain_values_pass_through(self) -> None:
        from src.core.instantiate import instantiate

        assert instantiate(42) == 42
        assert instantiate("hello") == "hello"
        assert instantiate([1, 2, 3]) == [1, 2, 3]


class TestBuildTransforms:
    def test_all_stages_present(self) -> None:
        config = load_config(_minimal_config())
        transforms = build_transforms(config)
        assert set(transforms) == {Stage.TRAIN, Stage.VAL, Stage.TEST, Stage.PREDICT}

    def test_non_albumentations_transform_used_directly(self) -> None:
        from src.transforms.input import IdentityTransform

        raw = _minimal_config()
        raw["transforms"] = {
            "train": {"_target_": "src.transforms.input.IdentityTransform"},
            "val": {"_target_": "src.transforms.input.IdentityTransform"},
        }
        config = load_config(raw)
        transforms = build_transforms(config)
        assert isinstance(transforms[Stage.TRAIN], IdentityTransform)

    def test_transforms_produce_correct_shape(self) -> None:
        import torch

        from src.core.entities import Sample

        config = load_config(_minimal_config())
        transform = build_transforms(config)[Stage.TRAIN]
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        sample = Sample(inputs={"image": image})
        result = transform.apply(sample)
        assert result.inputs["image"].shape == torch.Size([3, 64, 64])

    def test_config_transforms_used_when_present(self) -> None:
        import torch

        from src.core.entities import Sample

        raw = _minimal_config()
        raw["transforms"] = {
            "train": {
                "_target_": "albumentations.Compose",
                "transforms": [
                    {"_target_": "albumentations.Resize", "height": 16, "width": 16},
                    {"_target_": "albumentations.Normalize"},
                    {"_target_": "albumentations.pytorch.ToTensorV2"},
                ],
            },
            "val": {
                "_target_": "albumentations.Compose",
                "transforms": [
                    {"_target_": "albumentations.Resize", "height": 16, "width": 16},
                    {"_target_": "albumentations.Normalize"},
                    {"_target_": "albumentations.pytorch.ToTensorV2"},
                ],
            },
        }
        config = load_config(raw)
        transforms = build_transforms(config)
        assert set(transforms) >= {Stage.TRAIN, Stage.VAL}
        # test and predict derived from val
        assert Stage.TEST in transforms
        assert Stage.PREDICT in transforms

        image = np.zeros((128, 128, 3), dtype=np.uint8)
        sample = Sample(inputs={"image": image})
        result = transforms[Stage.TRAIN].apply(sample)
        assert result.inputs["image"].shape == torch.Size([3, 16, 16])


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
    def test_classification_gets_label_index_codec(self) -> None:
        config = load_config(_minimal_config())
        bindings = build_bindings(config)
        assert len(bindings) == 1
        assert isinstance(bindings[0].codec, LabelIndexCodec)
        assert bindings[0].column == "label"

    def test_regression_gets_float_codec(self) -> None:
        raw = _minimal_config()
        raw["tasks"] = {"score": {"preset": "regression", "target": "score", "dim": 1}}
        config = load_config(raw)
        bindings = build_bindings(config)
        assert isinstance(bindings[0].codec, FloatCodec)

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
        assert isinstance(bindings[0].codec, MultiLabelBinarizeCodec)

    def test_custom_separator_via_target_codec_spec(self) -> None:
        raw = _minimal_config()
        raw["tasks"] = {
            "tags": {
                "preset": "classification",
                "target": "tags",
                "objective": "multilabel",
                "target_codec": {"name": "multilabel_binarize", "separator": "|"},
            }
        }
        config = load_config(raw)
        bindings = build_bindings(config)
        assert isinstance(bindings[0].codec, MultiLabelBinarizeCodec)
        assert bindings[0].codec._separator == "|"

    def test_target_codec_override(self) -> None:
        raw = _minimal_config()
        raw["tasks"]["label"]["target_codec"] = "label_index"
        config = load_config(raw)
        bindings = build_bindings(config)
        assert isinstance(bindings[0].codec, LabelIndexCodec)

    def test_segmentation_gets_mask_codec(self) -> None:
        from src.data.codecs import MaskCodec

        raw = _minimal_config()
        raw["tasks"] = {"mask": {"preset": "segmentation", "target": "mask_path", "num_classes": 4}}
        config = load_config(raw)
        bindings = build_bindings(config)
        assert isinstance(bindings[0].codec, MaskCodec)


class TestDataConfigModes:
    def test_split_mode_str(self) -> None:
        config = load_config(_minimal_config())
        assert config.data.sources == "data.csv"
        assert config.data.split is not None

    def test_split_mode_list(self) -> None:
        config = load_config(
            _minimal_config(
                data={
                    "sources": ["a.csv", "b.csv"],
                    "inputs": "image_path",
                    "split": {"train": 0.8, "val": 0.1, "test": 0.1},
                }
            )
        )
        assert isinstance(config.data.sources, list)

    def test_presplit_mode_dict(self) -> None:
        config = load_config(
            _minimal_config(
                data={
                    "sources": {"train": "train.csv", "val": "val.csv"},
                    "inputs": "image_path",
                }
            )
        )
        assert isinstance(config.data.sources, dict)
        assert config.data.split is None

    def test_presplit_list_of_paths_per_stage(self) -> None:
        config = load_config(
            _minimal_config(
                data={
                    "sources": {"train": ["a.csv", "b.csv"], "val": "val.csv"},
                    "inputs": "image_path",
                }
            )
        )
        assert isinstance(config.data.sources, dict)
        assert isinstance(config.data.sources["train"], list)

    def test_presplit_with_split_raises(self) -> None:
        with pytest.raises(Exception, match="split"):
            load_config(
                _minimal_config(
                    data={
                        "sources": {"train": "train.csv", "val": "val.csv"},
                        "inputs": "image_path",
                        "split": {"train": 0.8, "val": 0.1, "test": 0.1},
                    }
                )
            )

    def test_split_mode_without_split_raises(self) -> None:
        with pytest.raises(Exception, match="split"):
            load_config(
                _minimal_config(
                    data={
                        "sources": "data.csv",
                        "inputs": "image_path",
                    }
                )
            )

    def test_presplit_missing_train_raises(self) -> None:
        with pytest.raises(Exception, match="train"):
            load_config(
                _minimal_config(
                    data={
                        "sources": {"val": "val.csv"},
                        "inputs": "image_path",
                    }
                )
            )

    def test_invalid_stage_key_raises(self) -> None:
        with pytest.raises(Exception):
            load_config(
                _minimal_config(
                    data={
                        "sources": {"train": "train.csv", "predict": "pred.csv"},
                        "inputs": "image_path",
                    }
                )
            )


class TestBuildDataSource:
    def test_infers_csv_from_extension(self) -> None:
        config = load_config(
            _minimal_config(
                data={
                    "sources": "data/x.csv",
                    "inputs": "image_path",
                    "split": {"train": 0.8, "val": 0.1, "test": 0.1},
                }
            )
        )
        assert isinstance(build_data_source(config.data), CsvDataSource)

    def test_infers_json_from_extension(self) -> None:
        config = load_config(
            _minimal_config(
                data={
                    "sources": "data/x.json",
                    "inputs": "image_path",
                    "split": {"train": 0.8, "val": 0.1, "test": 0.1},
                }
            )
        )
        assert isinstance(build_data_source(config.data), JsonDataSource)

    def test_explicit_source_type_overrides_extension(self) -> None:
        config = load_config(
            _minimal_config(
                data={
                    "sources": "data/x.dat",
                    "source_type": "csv",
                    "inputs": "image_path",
                    "split": {"train": 0.8, "val": 0.1, "test": 0.1},
                }
            )
        )
        assert isinstance(build_data_source(config.data), CsvDataSource)

    def test_unknown_extension_raises(self) -> None:
        config = load_config(
            _minimal_config(
                data={
                    "sources": "data/x.parquet",
                    "inputs": "image_path",
                    "split": {"train": 0.8, "val": 0.1, "test": 0.1},
                }
            )
        )
        with pytest.raises(ValueError, match="Unknown source extension"):
            build_data_source(config.data)

    def test_mixed_extensions_raise(self) -> None:
        config = load_config(
            _minimal_config(
                data={
                    "sources": ["a.csv", "b.json"],
                    "inputs": "image_path",
                    "split": {"train": 0.8, "val": 0.1, "test": 0.1},
                }
            )
        )
        with pytest.raises(ValueError, match="mixed extensions"):
            build_data_source(config.data)

    def test_multitask_order_preserved(self) -> None:
        raw = _minimal_config()
        raw["tasks"] = {
            "species": {"preset": "classification", "target": "species"},
            "age": {"preset": "classification", "target": "age"},
        }
        config = load_config(raw)
        runtime = RuntimeContext(num_classes={"species": 3, "age": 4})
        tasks = build_tasks(config, runtime)
        assert [t.name for t in tasks] == ["species", "age"]
        assert tasks[0].head_spec.out_features == 3
        assert tasks[1].head_spec.out_features == 4


class TestBuildTaskLrOverrides:
    def test_no_overrides_returns_empty(self) -> None:
        config = load_config(_minimal_config())
        assert build_task_lr_overrides(config) == {}

    def test_single_task_override(self) -> None:
        raw = _minimal_config()
        raw["tasks"]["label"]["optimizer"] = {"lr": 1e-4}
        config = load_config(raw)
        overrides = build_task_lr_overrides(config)
        assert overrides == {"label": pytest.approx(1e-4)}

    def test_partial_override_only_includes_declared_tasks(self) -> None:
        raw = _minimal_config()
        raw["tasks"]["species"] = {"preset": "classification", "target": "species", "optimizer": {"lr": 5e-5}}
        raw["tasks"]["age"] = {"preset": "classification", "target": "age"}
        config = load_config(raw)
        overrides = build_task_lr_overrides(config)
        assert set(overrides.keys()) == {"species"}
        assert overrides["species"] == pytest.approx(5e-5)


class TestBuildLitModule:
    def test_creates_lit_module_without_overrides(self) -> None:
        from src.models import build_composite_model
        from src.models.registry import backbones
        from src.training import LitModule, OptimizerBuilder

        config = load_config(_minimal_config())
        runtime = RuntimeContext(num_classes={"label": 3})
        tasks = build_tasks(config, runtime)
        backbone = backbones.create("timm", name="resnet18", pretrained=False)
        model = build_composite_model(backbone, {t.name: t.head_spec for t in tasks})
        opt_builder = OptimizerBuilder(base_lr=1e-3)

        lit = build_lit_module(config, model, tasks, opt_builder)
        assert isinstance(lit, LitModule)
        assert lit._task_lr_overrides == {}

    def test_creates_lit_module_with_per_task_lr(self) -> None:
        from src.models import build_composite_model
        from src.models.registry import backbones
        from src.training import LitModule, OptimizerBuilder

        raw = _minimal_config()
        raw["tasks"]["label"]["optimizer"] = {"lr": 1e-4}
        config = load_config(raw)
        runtime = RuntimeContext(num_classes={"label": 3})
        tasks = build_tasks(config, runtime)
        backbone = backbones.create("timm", name="resnet18", pretrained=False)
        model = build_composite_model(backbone, {t.name: t.head_spec for t in tasks})
        opt_builder = OptimizerBuilder(base_lr=1e-3)

        lit = build_lit_module(config, model, tasks, opt_builder)
        assert isinstance(lit, LitModule)
        assert lit._task_lr_overrides == {"label": pytest.approx(1e-4)}

    def test_param_groups_reflect_per_task_lr(self) -> None:
        from src.models import build_composite_model
        from src.models.registry import backbones
        from src.training import OptimizerBuilder

        raw = _minimal_config()
        raw["tasks"]["label"]["optimizer"] = {"lr": 1e-4}
        config = load_config(raw)
        runtime = RuntimeContext(num_classes={"label": 3})
        tasks = build_tasks(config, runtime)
        backbone = backbones.create("timm", name="resnet18", pretrained=False)
        model = build_composite_model(backbone, {t.name: t.head_spec for t in tasks})
        opt_builder = OptimizerBuilder(base_lr=1e-3)

        lit = build_lit_module(config, model, tasks, opt_builder)
        result = lit.configure_optimizers()
        assert isinstance(result, torch.optim.Optimizer)
        lrs = {g["name"]: g["lr"] for g in result.param_groups}
        assert lrs["backbone"] == pytest.approx(1e-3)
        assert lrs["head/label"] == pytest.approx(1e-4)


class TestFullWiringSmoke:
    """Wires all components from a config dict (no Hydra, no Trainer)."""

    @pytest.fixture
    def csv_path(self, tmp_path: Path) -> Path:
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        rng = np.random.default_rng(2)
        rows = []
        for i in range(20):
            arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            p = image_dir / f"{i}.jpg"
            cv2.imwrite(str(p), arr)
            rows.append({"image_path": str(p), "label": ["cat", "dog", "cow"][i % 3]})
        csv = tmp_path / "data.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        return csv

    def test_end_to_end_wiring(self, csv_path: Path) -> None:
        from src.data import CsvDataSource, DataModule
        from src.models import build_composite_model
        from src.models.registry import backbones
        from src.training import LitModule, OptimizerBuilder

        raw = _minimal_config()
        raw["data"]["sources"] = str(csv_path)
        config = load_config(raw)

        runtime = RuntimeContext()
        plain_dm = DataModule(
            target_bindings=build_bindings(config),
            inputs_config="image_path",
            transforms=build_transforms(config),
            runtime=runtime,
            batch_size=4,
            seed=0,
            source=CsvDataSource([str(csv_path)]),
            split=config.data.split,
        )
        plain_dm.setup()

        tasks = build_tasks(config, runtime)
        backbone = backbones.create("timm", name="resnet18", pretrained=False)
        model = build_composite_model(backbone, {t.name: t.head_spec for t in tasks})
        lit = LitModule(model=model, tasks=tasks, optimizer_builder=OptimizerBuilder(1e-3))

        import torch

        batch = next(iter(plain_dm.train_dataloader()))
        result = lit.training_step(batch, 0)
        assert isinstance(result["loss"], torch.Tensor)
        assert result["loss"].ndim == 0


class TestBuildLogger:
    def test_none_kind_returns_false(self) -> None:
        config = load_config(_minimal_config())
        assert config.logger.kind == "none"
        assert build_logger(config) is False

    def test_default_logger_config_is_none(self) -> None:
        from src.config.schema import LoggerConfig

        cfg = LoggerConfig()
        assert cfg.kind == "none"
        assert cfg.project is None
        assert cfg.task is None

    def test_unknown_kind_raises(self) -> None:
        raw = _minimal_config()
        raw["logger"] = {"kind": "wandb"}
        config = load_config(raw)
        with pytest.raises(ValueError, match="Unknown logger kind"):
            build_logger(config)

    def test_logger_config_parses_clearml_kind(self) -> None:
        raw = _minimal_config()
        raw["logger"] = {"kind": "clearml", "project": "ml-tests"}
        config = load_config(raw)
        assert config.logger.kind == "clearml"
        assert config.logger.project == "ml-tests"

    def test_clearml_logger_is_both_instances(self) -> None:
        """ClearMLLogger must satisfy both Lightning Logger and PlotLogger."""
        pytest.importorskip("clearml")
        from unittest.mock import MagicMock, patch

        import lightning as L

        from src.core.ports import PlotLogger
        from src.loggers.clearml import ClearMLLogger

        mock_task = MagicMock()
        mock_task.name = "test-task"
        mock_task.id = "abc-123"
        mock_task.get_logger.return_value = MagicMock()

        with patch("clearml.Task.init", return_value=mock_task):
            logger = ClearMLLogger(project_name="test-proj", task_name="test-task")

        assert isinstance(logger, PlotLogger)
        assert isinstance(logger, L.pytorch.loggers.Logger)

    def test_clearml_log_html_reports_media(self) -> None:
        pytest.importorskip("clearml")
        from unittest.mock import MagicMock, patch

        from src.loggers.clearml import ClearMLLogger

        mock_task = MagicMock()
        mock_backend = MagicMock()
        mock_task.name = "t"
        mock_task.id = "i"
        mock_task.get_logger.return_value = mock_backend
        with patch("clearml.Task.init", return_value=mock_task):
            logger = ClearMLLogger(project_name="p", task_name="t")

        logger.log_html("samples/val", "<html><body>hello</body></html>", iteration=2)
        mock_backend.report_media.assert_called_once()
        kwargs = mock_backend.report_media.call_args.kwargs
        assert kwargs["title"] == "samples/val"
        assert kwargs["iteration"] == 2
        assert kwargs["file_extension"] == "html"
        assert "hello" in kwargs["stream"].getvalue()

    def test_clearml_split_metric_name_multipart(self) -> None:
        pytest.importorskip("clearml")
        from src.loggers.clearml import ClearMLLogger

        title, series = ClearMLLogger._split_metric_name("label/f1/val")
        assert title == "label/f1"
        assert series == "val"

    def test_clearml_split_metric_name_single(self) -> None:
        pytest.importorskip("clearml")
        from src.loggers.clearml import ClearMLLogger

        title, series = ClearMLLogger._split_metric_name("loss")
        assert title == "loss"
        assert series == "value"

    def test_logger_config_parses_tags(self) -> None:
        raw = _minimal_config()
        raw["logger"] = {"kind": "clearml", "tags": ["timm", "resnet18", "lr=0.001"]}
        config = load_config(raw)
        assert config.logger.tags == ["timm", "resnet18", "lr=0.001"]

    def test_logger_config_drops_empty_tags(self) -> None:
        """A None tag (e.g. ${backbone.name} on a multi backbone) or empty string is dropped."""
        raw = _minimal_config()
        raw["logger"] = {"kind": "clearml", "tags": ["timm", None, "", "lr=0.001"]}
        config = load_config(raw)
        assert config.logger.tags == ["timm", "lr=0.001"]

    def test_clearml_builder_forwards_tags(self) -> None:
        pytest.importorskip("clearml")
        from unittest.mock import MagicMock, patch

        raw = _minimal_config()
        raw["logger"] = {"kind": "clearml", "tags": ["timm", "lr=0.001"]}
        config = load_config(raw)

        mock_task = MagicMock()
        mock_task.name = "t"
        mock_task.id = "i"
        mock_task.get_logger.return_value = MagicMock()
        with patch("clearml.Task.init", return_value=mock_task) as init:
            build_logger(config)
        assert init.call_args.kwargs["tags"] == ["timm", "lr=0.001"]


class TestBuildBatchTransform:
    def test_mixup_built_with_global_targets(self) -> None:
        from src.composition.wiring import WiringContext
        from src.composition.wiring.callbacks import _build_batch_transform
        from src.transforms.batch import MixUp

        config = load_config(_minimal_config())  # 'label' = classification (GLOBAL)
        runtime = RuntimeContext()
        runtime.num_classes["label"] = 3
        ctx = WiringContext(config=config, runtime=runtime)

        transform = _build_batch_transform({"name": "mixup", "alpha": 0.2}, ctx)

        assert isinstance(transform, MixUp)
        assert transform._targets[0].num_classes == 3  # injected from the task

    def test_mosaic_built_with_dense_targets(self) -> None:
        from src.composition.wiring import WiringContext
        from src.composition.wiring.callbacks import _build_batch_transform
        from src.transforms.batch import Mosaic

        raw = _minimal_config()
        raw["tasks"] = {"mask": {"preset": "segmentation", "target": "mask_path", "num_classes": 4}}
        config = load_config(raw)
        ctx = WiringContext(config=config, runtime=RuntimeContext())

        transform = _build_batch_transform({"name": "mosaic"}, ctx)

        assert isinstance(transform, Mosaic)

    def test_guard_rejects_mixup_with_dense_head(self) -> None:
        from src.composition.wiring import WiringContext
        from src.composition.wiring.callbacks import _build_batch_transform

        raw = _minimal_config()
        raw["tasks"] = {"mask": {"preset": "segmentation", "target": "mask_path", "num_classes": 4}}
        config = load_config(raw)
        ctx = WiringContext(config=config, runtime=RuntimeContext())

        with pytest.raises(ValueError, match="coherent target"):
            _build_batch_transform({"name": "mixup"}, ctx)

    def test_build_callbacks_wires_batch_transform(self) -> None:
        from src.composition.wiring import build_callbacks

        raw = _minimal_config()
        raw["callbacks"] = {
            "mixup": {"name": "batch_transform", "disable_after_fraction": 0.5, "transform": {"name": "mixup"}}
        }
        config = load_config(raw)
        runtime = RuntimeContext()
        runtime.num_classes["label"] = 3

        callbacks = build_callbacks(config, runtime)

        assert len(callbacks) == 1
        assert isinstance(callbacks[0], BatchTransformCallback)
