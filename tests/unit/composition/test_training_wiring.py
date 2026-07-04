"""Training-layer wiring: LitModule, logger, trainer, batch-transform guard, per-task LR."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from src.callbacks.batch_transform import BatchTransformCallback
from src.composition.wiring import (
    build_lit_module,
    build_logger,
    build_task_lr_overrides,
    build_tasks,
)
from src.config import load_config
from src.core.runtime import RuntimeContext
from tests.support.builders import minimal_config as _minimal_config


class TestBuildLitModule:
    def test_creates_lit_module_and_serialises_hparams(self) -> None:
        # Per-task LR lives on the OptimizerBuilder now (see TestOptimizerBuilder); build_lit_module
        # only wires collaborators + hparams, so it no longer branches on per-task overrides.
        from src.models import build_composite_model
        from src.models.registry import backbones
        from src.training import LitModule, OptimizerBuilder

        config = load_config(_minimal_config())
        runtime = RuntimeContext(num_classes={"label": 3})
        tasks = build_tasks(config, runtime)
        backbone = backbones.create("timm", name="resnet18", pretrained=False)
        model = build_composite_model(backbone, {t.name: t.head_spec for t in tasks})

        lit = build_lit_module(config, model, tasks, OptimizerBuilder(base_lr=1e-3))
        assert isinstance(lit, LitModule)
        assert lit._hparams_to_log is not None  # full config serialised as hyperparams

    def test_param_groups_reflect_per_task_lr(self) -> None:
        from src.composition.wiring import build_optimizer_builder
        from src.models import build_composite_model
        from src.models.registry import backbones

        raw = _minimal_config()
        raw["tasks"]["label"]["optimizer"] = {"lr": 1e-4}
        config = load_config(raw)
        runtime = RuntimeContext(num_classes={"label": 3})
        tasks = build_tasks(config, runtime)
        backbone = backbones.create("timm", name="resnet18", pretrained=False)
        model = build_composite_model(backbone, {t.name: t.head_spec for t in tasks})
        opt_builder = build_optimizer_builder(config.optimizer, build_task_lr_overrides(config))

        lit = build_lit_module(config, model, tasks, opt_builder)
        result = lit.configure_optimizers()
        assert isinstance(result, torch.optim.Optimizer)
        lrs = {g["name"]: g["lr"] for g in result.param_groups}
        assert lrs["backbone"] == pytest.approx(1e-3)
        assert lrs["head/label"] == pytest.approx(1e-4)


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
        """ClearMLLogger must satisfy Lightning's Logger and every artifact-logger port."""
        pytest.importorskip("clearml")
        from unittest.mock import MagicMock, patch

        import lightning as L

        from src.core.ports import (
            CurveLogger,
            HistogramLogger,
            HtmlLogger,
            MatrixLogger,
            PlotLogger,
            SingleValueLogger,
        )
        from src.loggers.clearml import ClearMLLogger

        mock_task = MagicMock()
        mock_task.name = "test-task"
        mock_task.id = "abc-123"
        mock_task.get_logger.return_value = MagicMock()

        with patch("clearml.Task.init", return_value=mock_task):
            logger = ClearMLLogger(project_name="test-proj", task_name="test-task")

        assert isinstance(logger, L.pytorch.loggers.Logger)
        for port in (MatrixLogger, CurveLogger, HtmlLogger, SingleValueLogger, HistogramLogger, PlotLogger):
            assert isinstance(logger, port)

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


class TestBuildTrainer:
    """``build_trainer`` is the single home for Trainer construction (profiler seam)."""

    @staticmethod
    def _cpu_trainer(**trainer_extra: Any) -> dict[str, Any]:
        return {"accelerator": "cpu", "devices": 1, **trainer_extra}

    def test_profiler_target_dict_is_instantiated(self) -> None:
        from lightning.pytorch.profilers import AdvancedProfiler

        from src.composition.wiring import build_trainer

        config = load_config(
            _minimal_config(
                trainer=self._cpu_trainer(
                    profiler={
                        "_target_": "lightning.pytorch.profilers.AdvancedProfiler",
                        "dirpath": "/tmp/prof",
                        "filename": "report",
                    }
                )
            )
        )
        trainer = build_trainer(config, logger=False, callbacks=[])
        assert isinstance(getattr(trainer, "profiler"), AdvancedProfiler)

    def test_profiler_string_alias_passes_through(self) -> None:
        from lightning.pytorch.profilers import SimpleProfiler

        from src.composition.wiring import build_trainer

        config = load_config(_minimal_config(trainer=self._cpu_trainer(profiler="simple")))
        trainer = build_trainer(config, logger=False, callbacks=[])
        assert isinstance(getattr(trainer, "profiler"), SimpleProfiler)

    def test_profiler_none_disables_profiling(self) -> None:
        from lightning.pytorch.profilers import PassThroughProfiler

        from src.composition.wiring import build_trainer

        config = load_config(_minimal_config(trainer=self._cpu_trainer()))
        trainer = build_trainer(config, logger=False, callbacks=[])
        assert isinstance(getattr(trainer, "profiler"), PassThroughProfiler)

    def test_epochs_and_save_dir_forwarded(self) -> None:
        from src.composition.wiring import build_trainer

        config = load_config(_minimal_config(epochs=7, save_dir="/tmp/run", trainer=self._cpu_trainer()))
        trainer = build_trainer(config, logger=False, callbacks=[])
        assert trainer.max_epochs == 7
        assert str(trainer.default_root_dir) == "/tmp/run"

    def test_dataloader_block_maps_to_config(self) -> None:
        config = load_config(_minimal_config(dataloader={"num_workers": 3, "pin_memory": True}))
        assert config.dataloader.num_workers == 3
        assert config.dataloader.pin_memory is True

    def test_dataloader_extra_key_survives_validation(self) -> None:
        config = load_config(_minimal_config(dataloader={"timeout": 5}))
        assert config.dataloader.model_extra == {"timeout": 5}

    def test_dataloader_reserved_key_is_rejected(self) -> None:
        from src.config import ConfigError

        with pytest.raises(ConfigError, match="managed by the framework"):
            load_config(_minimal_config(dataloader={"shuffle": True}))


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
