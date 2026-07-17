"""EmaModelCheckpoint: weights-only checkpoints stay consistent with the EMA weights.

Lightning's ``dump_checkpoint`` invokes callbacks' ``on_save_checkpoint`` only when
``weights_only=False``, so with ``save_weights_only=True`` a plain ``ModelCheckpoint``
stores the live weights while the monitored metric came from the averaged ones.
``EmaModelCheckpoint`` re-applies the EMA hook on that path and defers to the parent
everywhere else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.callbacks.ema import EmaCallback
from src.callbacks.ema_checkpoint import EmaModelCheckpoint
from src.callbacks.registry import callback_registry
from tests.support.fakes import TinyLitModule, make_mock_trainer

# ---------------------------------------------------------------- helpers

LIVE = torch.full((2, 2), 2.0)  # weights after manual divergence
AVERAGED = torch.ones(2, 2)  # weights captured by the EMA average at init


def _prepared_ema(module: TinyLitModule, latest_update_step: int) -> EmaCallback:
    """EMA whose averaged model holds the init weights (ones); live model diverged to twos."""
    ema = EmaCallback(decay=0.999, warmup_fraction=0.5)
    ema.setup(make_mock_trainer(estimated_stepping_batches=200), module, stage="fit")
    with torch.no_grad():
        module.linear.weight.copy_(LIVE)
    ema._latest_update_step = latest_update_step
    return ema


def _saving_trainer(module: TinyLitModule, *callbacks: L.Callback) -> MagicMock:
    """Mock trainer wired for the ``_save_checkpoint`` path (dump → hook → strategy save)."""
    trainer = make_mock_trainer(global_step=7)
    trainer.callbacks = list(callbacks)
    trainer.lightning_module = module
    trainer.loggers = []
    trainer.is_global_zero = True
    trainer._checkpoint_connector.dump_checkpoint.return_value = {
        "state_dict": {key: value.clone() for key, value in module.state_dict().items()}
    }
    return trainer


def _saved_checkpoint(trainer: MagicMock) -> dict[str, Any]:
    """The checkpoint dict handed to ``strategy.save_checkpoint``."""
    return cast("dict[str, Any]", trainer.strategy.save_checkpoint.call_args.args[0])


# ------------------------------------------- weights-only path: EMA hook applied


class TestWeightsOnlySavesAveragedWeights:
    def test_state_dict_replaced_with_averaged_weights(self, tmp_path: Path) -> None:
        module = TinyLitModule()
        ema = _prepared_ema(module, latest_update_step=5)
        checkpoint_callback = EmaModelCheckpoint(dirpath=tmp_path, save_weights_only=True)
        trainer = _saving_trainer(module, ema, checkpoint_callback)

        checkpoint_callback._save_checkpoint(trainer, str(tmp_path / "best.ckpt"))

        trainer._checkpoint_connector.dump_checkpoint.assert_called_once_with(weights_only=True)
        saved = _saved_checkpoint(trainer)
        assert torch.allclose(saved["state_dict"]["linear.weight"], AVERAGED)
        assert torch.allclose(saved["current_model_state"]["linear.weight"], LIVE)

    def test_warmup_checkpoint_keeps_live_weights(self, tmp_path: Path) -> None:
        """Before averaging starts the EMA hook's own guard applies: live weights, no EMA copy."""
        module = TinyLitModule()
        ema = _prepared_ema(module, latest_update_step=0)
        checkpoint_callback = EmaModelCheckpoint(dirpath=tmp_path, save_weights_only=True)
        trainer = _saving_trainer(module, ema, checkpoint_callback)

        checkpoint_callback._save_checkpoint(trainer, str(tmp_path / "best.ckpt"))

        saved = _saved_checkpoint(trainer)
        assert torch.allclose(saved["state_dict"]["linear.weight"], LIVE)
        assert "current_model_state" not in saved

    def test_bookkeeping_mirrors_parent(self, tmp_path: Path) -> None:
        """The parent's save bookkeeping (last step/path) must also happen on the EMA path."""
        module = TinyLitModule()
        ema = _prepared_ema(module, latest_update_step=5)
        checkpoint_callback = EmaModelCheckpoint(dirpath=tmp_path, save_weights_only=True)
        trainer = _saving_trainer(module, ema, checkpoint_callback)
        filepath = str(tmp_path / "best.ckpt")

        checkpoint_callback._save_checkpoint(trainer, filepath)

        assert checkpoint_callback._last_global_step_saved == trainer.global_step
        assert checkpoint_callback._last_checkpoint_saved == filepath


# ------------------------------------------------- parent path: no intervention


class TestParentPathDelegation:
    def test_full_checkpoint_defers_to_parent(self, tmp_path: Path) -> None:
        """With save_weights_only=False Lightning calls the EMA hook itself — no intervention."""
        module = TinyLitModule()
        ema = _prepared_ema(module, latest_update_step=5)
        checkpoint_callback = EmaModelCheckpoint(dirpath=tmp_path, save_weights_only=False)
        trainer = _saving_trainer(module, ema, checkpoint_callback)
        filepath = str(tmp_path / "best.ckpt")

        checkpoint_callback._save_checkpoint(trainer, filepath)

        trainer.save_checkpoint.assert_called_once_with(filepath, False)
        trainer.strategy.save_checkpoint.assert_not_called()

    def test_weights_only_without_ema_defers_to_parent(self, tmp_path: Path) -> None:
        module = TinyLitModule()
        checkpoint_callback = EmaModelCheckpoint(dirpath=tmp_path, save_weights_only=True)
        trainer = _saving_trainer(module, checkpoint_callback)
        filepath = str(tmp_path / "best.ckpt")

        checkpoint_callback._save_checkpoint(trainer, filepath)

        trainer.save_checkpoint.assert_called_once_with(filepath, True)
        trainer.strategy.save_checkpoint.assert_not_called()

    def test_stock_ema_weight_averaging_is_recognized(self, tmp_path: Path) -> None:
        """The EMA lookup matches Lightning's base EMAWeightAveraging, not only our subclass.

        Regression: an EmaCallback-narrow isinstance let a run configured with the stock
        callback silently fall back to live weights in weights-only checkpoints.
        """
        from lightning.pytorch.callbacks import EMAWeightAveraging

        module = TinyLitModule()
        stock_ema = EMAWeightAveraging()
        stock_ema.setup(make_mock_trainer(), module, stage="fit")
        checkpoint_callback = EmaModelCheckpoint(dirpath=tmp_path, save_weights_only=True)
        trainer = _saving_trainer(module, stock_ema, checkpoint_callback)

        checkpoint_callback._save_checkpoint(trainer, str(tmp_path / "best.ckpt"))

        # The EMA path ran (dump + manual hook + strategy save), not the parent delegation.
        trainer._checkpoint_connector.dump_checkpoint.assert_called_once_with(weights_only=True)
        trainer.save_checkpoint.assert_not_called()


# ---------------------------------------------------------------- registration


class TestRegistration:
    def test_registered_under_checkpoint_key(self, tmp_path: Path) -> None:
        """The ``checkpoint`` registry key builds the EMA-aware subclass (config unchanged)."""
        import src.callbacks  # noqa: F401 — importing the package populates the registry

        callback = callback_registry.create("checkpoint", dirpath=str(tmp_path))
        assert isinstance(callback, EmaModelCheckpoint)


# --------------------------- Lightning private-API canary (real save path)
#
# The EMA path reads ``trainer._checkpoint_connector.dump_checkpoint`` and mirrors the
# parent's ``_last_*`` bookkeeping. The mock tests above set those names themselves, so
# a Lightning rename would slip past them. This canary drives a real fit through the
# real connector/strategy: a rename on a version bump fails loudly here instead of
# silently persisting live weights in production.


class _FitModule(L.LightningModule):
    """Trains for real (loss depends on the weights) so live and EMA weights diverge."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)

    def training_step(self, batch: list[torch.Tensor], batch_index: int) -> torch.Tensor:
        (features,) = batch
        output: torch.Tensor = self.linear(features)
        return output.pow(2).sum()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


class TestLightningPrivateApiCanary:
    def test_real_fit_writes_averaged_weights_into_weights_only_checkpoint(self, tmp_path: Path) -> None:
        L.seed_everything(0, workers=True, verbose=False)
        module = _FitModule()
        checkpoint_callback = EmaModelCheckpoint(dirpath=tmp_path, filename="best", save_weights_only=True)
        trainer = L.Trainer(
            max_epochs=2,
            accelerator="cpu",
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            callbacks=[EmaCallback(decay=0.5, warmup_fraction=0.0), checkpoint_callback],
        )
        trainer.fit(module, DataLoader(TensorDataset(torch.randn(8, 2)), batch_size=4))

        saved = torch.load(checkpoint_callback.best_model_path, weights_only=False)
        # on_train_end copied the EMA weights into the module — the file must hold the same.
        assert torch.equal(saved["state_dict"]["linear.weight"], module.linear.weight.detach())
        # The EMA hook really ran: the live weights were kept aside and differ from the average.
        assert not torch.equal(saved["current_model_state"]["linear.weight"], saved["state_dict"]["linear.weight"])
