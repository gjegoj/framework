"""Unit tests for fit/test orchestration and checkpoint resolution."""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.composition.wiring.checkpointing import load_init_weights, resolve_test_ckpt_path
from src.composition.wiring.training import run_experiment
from src.config import ExperimentConfig, load_config
from src.models.assembly import build_composite_model
from src.models.backbones import EmbeddingBackbone
from src.tasks import classification
from src.training.modules import LitModule
from src.training.optim import OptimizerBuilder
from tests.support.builders import raw_config as _raw


class _ArbitraryPayload:
    """A non-tensor object; module-level so it is picklable into a test checkpoint."""


class TestResolveTestCkptPath:
    def _config(self, **overrides: object) -> ExperimentConfig:
        return load_config(_raw(**overrides))

    def test_explicit_ckpt_path_wins(self) -> None:
        trainer = MagicMock()
        trainer.checkpoint_callback = MagicMock(best_model_path="/tmp/best.ckpt")
        config = self._config(ckpt_path="/explicit.ckpt")
        assert resolve_test_ckpt_path(trainer, config, trained=True) == "/explicit.ckpt"

    def test_best_after_training_when_checkpoint_saved(self) -> None:
        trainer = MagicMock()
        trainer.checkpoint_callback = MagicMock(best_model_path="/tmp/best.ckpt")
        config = self._config()
        assert resolve_test_ckpt_path(trainer, config, trained=True) == "best"

    def test_in_memory_when_no_checkpoint_callback(self) -> None:
        trainer = MagicMock()
        trainer.checkpoint_callback = None
        config = self._config()
        assert resolve_test_ckpt_path(trainer, config, trained=True) is None

    def test_in_memory_when_best_path_empty(self) -> None:
        trainer = MagicMock()
        trainer.checkpoint_callback = MagicMock(best_model_path="")
        config = self._config()
        assert resolve_test_ckpt_path(trainer, config, trained=True) is None


def _tasks() -> list:
    task = classification("label", num_classes=3)
    return [task]


class TestRunFitAndTest:
    def test_train_and_test_calls_fit_then_test_with_best(self) -> None:
        config = load_config(_raw())
        trainer = MagicMock()
        trainer.checkpoint_callback = MagicMock(best_model_path="/tmp/best.ckpt")
        lit_module = MagicMock()
        lit_dm = MagicMock()

        with patch("src.composition.wiring.training.run_export"):
            run_experiment(trainer, lit_module, lit_dm, config, _tasks())

        trainer.fit.assert_called_once_with(lit_module, lit_dm)
        # verbose=True: the mock trainer has no MetricsProgressBar, so Lightning prints its table.
        trainer.test.assert_called_once_with(lit_module, lit_dm, ckpt_path="best", verbose=True)

    def test_eval_only_skips_fit(self) -> None:
        config = load_config(_raw(run_train=False, ckpt_path="/weights.ckpt"))
        trainer = MagicMock()
        lit_module = MagicMock()
        lit_dm = MagicMock()

        with patch("src.composition.wiring.training.run_export"):
            run_experiment(trainer, lit_module, lit_dm, config, _tasks())

        trainer.fit.assert_not_called()
        trainer.test.assert_called_once_with(lit_module, lit_dm, ckpt_path="/weights.ckpt", verbose=True)

    def test_train_only_skips_test(self) -> None:
        config = load_config(_raw(run_test=False))
        trainer = MagicMock()
        lit_module = MagicMock()
        lit_dm = MagicMock()

        with patch("src.composition.wiring.training.run_export"):
            run_experiment(trainer, lit_module, lit_dm, config, _tasks())

        trainer.fit.assert_called_once()
        trainer.test.assert_not_called()

    def test_init_ckpt_loaded_before_fit(self) -> None:
        config = load_config(_raw(init_ckpt_path="/pretrain.ckpt"))
        trainer = MagicMock()
        trainer.checkpoint_callback = MagicMock(best_model_path="/tmp/best.ckpt")
        lit_module = MagicMock()
        lit_dm = MagicMock()

        with (
            patch("src.composition.wiring.training.load_init_weights") as mock_load,
            patch("src.composition.wiring.training.run_export"),
        ):
            run_experiment(trainer, lit_module, lit_dm, config, _tasks())

        mock_load.assert_called_once_with(lit_module, "/pretrain.ckpt")
        trainer.fit.assert_called_once_with(lit_module, lit_dm)


class TestLoadInitWeights:
    def _lit_module(self) -> LitModule:
        task = classification("label", num_classes=3)
        model = build_composite_model(EmbeddingBackbone(embedding_dim=8), {"label": task.head_spec})
        return LitModule(model=model, tasks=[task], optimizer_builder=OptimizerBuilder(base_lr=1e-3))

    def test_loads_lightning_checkpoint_state_dict(self, tmp_path: Path) -> None:
        source = self._lit_module()
        weight_key = "model.heads.label.fc.weight"
        original = source.state_dict()[weight_key].detach().clone()
        ckpt_path = tmp_path / "init.ckpt"
        torch.save({"state_dict": source.state_dict(), "epoch": 99}, ckpt_path)

        target = self._lit_module()
        assert not torch.allclose(target.state_dict()[weight_key], original)

        load_init_weights(target, str(ckpt_path))

        assert torch.allclose(target.state_dict()[weight_key], original)

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="init_ckpt_path not found"):
            load_init_weights(self._lit_module(), "/no/such/checkpoint.ckpt")

    def test_rejects_checkpoint_smuggling_arbitrary_object(self, tmp_path: Path) -> None:
        """weights_only=True: a ckpt carrying a non-tensor pickled object is refused (RCE guard).

        With weights_only=False torch happily unpickles the object and the state_dict loads; the
        safe loader must instead reject the file before any arbitrary code can run.
        """
        source = self._lit_module()
        ckpt_path = tmp_path / "smuggled.ckpt"
        torch.save({"state_dict": source.state_dict(), "payload": _ArbitraryPayload()}, ckpt_path)
        with pytest.raises(pickle.UnpicklingError):
            load_init_weights(self._lit_module(), str(ckpt_path))


class TestRunModeValidation:
    def test_all_disabled_raises(self) -> None:
        with pytest.raises(Exception, match="run_train, run_test, or run_export"):
            load_config(_raw(run_train=False, run_test=False, run_export=False))

    def test_eval_only_requires_ckpt_path(self) -> None:
        with pytest.raises(Exception, match="ckpt_path is required"):
            load_config(_raw(run_train=False, run_test=True))

    def test_export_only_requires_ckpt_path(self) -> None:
        with pytest.raises(Exception, match="ckpt_path is required"):
            load_config(_raw(run_train=False, run_test=False, run_export=True))

    def test_export_only_with_ckpt_path_valid(self) -> None:
        config = load_config(_raw(run_train=False, run_test=False, run_export=True, ckpt_path="/x.ckpt"))
        assert config.run_export is True

    def test_init_ckpt_requires_run_train(self) -> None:
        with pytest.raises(Exception, match="init_ckpt_path requires run_train"):
            load_config(_raw(run_train=False, run_test=True, ckpt_path="/x.ckpt", init_ckpt_path="/y.ckpt"))

    def test_defaults_train_and_test(self) -> None:
        config = load_config(_raw())
        assert config.run_train is True
        assert config.run_test is True
        assert config.run_export is True
        assert [t.format for t in config.export.targets] == ["onnx"]
        assert config.ckpt_path is None
