"""``EmaWeights``: when the average takes over from the live weights, and when it must not."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.callbacks import EmaModelCheckpoint, EmaWeights
from src.callbacks.registry import callback_registry

STEPS_PER_EPOCH = 2
INITIAL = 0.0
from tests.support.lightning import quiet_trainer


class Recorder(L.LightningModule):
    """One weight, driven away from zero, and a note of what validation saw.

    Starting at zero is what makes the untrained copy recognisable: the averaged
    model is a snapshot of these weights taken before training moved them.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.net.weight.fill_(INITIAL)
        self.validated: list[float] = []

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss: torch.Tensor = ((self.net(batch[0]) - 1.0) ** 2).mean()
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self.validated.append(self.net.weight.item())

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # Slow enough that the weight is still climbing when the run ends, so an
        # average that lags behind it is visibly different from it.
        return torch.optim.SGD(self.parameters(), lr=0.1)


def fit(
    ema: EmaWeights,
    epochs: int = 2,
    extra: list[L.Callback] | None = None,
    root: Path | None = None,
) -> tuple[Recorder, L.Trainer]:
    module = Recorder()
    data = DataLoader(TensorDataset(torch.ones(STEPS_PER_EPOCH, 1)), batch_size=1)
    # One validation batch per epoch, so an entry of ``validated`` is an epoch.
    checking = DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1)
    trainer = quiet_trainer(
        max_epochs=epochs,
        default_root_dir=root,
        enable_checkpointing=extra is not None,
        callbacks=[ema, *(extra or [])],
    )
    trainer.fit(module, data, checking)
    return module, trainer


def test_the_warmup_share_becomes_an_absolute_step() -> None:
    """A share survives a change of epoch count; an absolute step is what Lightning wants."""
    ema = EmaWeights(decay=0.9, after=0.5)

    fit(ema, epochs=2)

    assert ema.update_starting_at_step == STEPS_PER_EPOCH  # half of two epochs of two steps


def test_validation_during_warmup_sees_the_live_weights() -> None:
    """Lightning swaps unconditionally, which would validate the untrained copy."""
    ema = EmaWeights(decay=0.9, after=0.5)

    module, _ = fit(ema, epochs=2)

    assert module.validated[0] != pytest.approx(INITIAL)


def test_validation_after_warmup_sees_the_averaged_weights() -> None:
    """Which is the whole point: the reported metric describes the model being kept."""
    ema = EmaWeights(decay=0.9, after=0.5)

    module, _ = fit(ema, epochs=2)

    assert module.validated[1] != pytest.approx(module.validated[0])


def test_a_checkpoint_saved_during_warmup_keeps_the_live_state() -> None:
    """Otherwise a 'best' checkpoint holds untrained weights while its metric came from live ones."""
    ema = EmaWeights(decay=0.9)
    module = Recorder()
    checkpoint: dict[str, Any] = {"state_dict": {"net.weight": torch.tensor([[3.0]])}}
    ema.setup(quiet_trainer(), module, "fit")

    ema.on_save_checkpoint(quiet_trainer(), module, checkpoint)

    assert "current_model_state" not in checkpoint
    assert checkpoint["state_dict"]["net.weight"].item() == 3.0


def test_a_checkpoint_saved_after_averaging_holds_the_average() -> None:
    ema = EmaWeights(decay=0.9)
    module, trainer = fit(ema, epochs=1)
    checkpoint: dict[str, Any] = {"state_dict": {"net.weight": torch.tensor([[3.0]])}}

    ema.on_save_checkpoint(trainer, module, checkpoint)

    assert "current_model_state" in checkpoint
    assert checkpoint["state_dict"]["net.weight"].item() != 3.0


def test_a_run_whose_warmup_never_ends_keeps_its_trained_weights() -> None:
    """Lightning would copy the untrained snapshot over the trained model at the end."""
    ema = EmaWeights(decay=0.9, after=0.99)

    module, _ = fit(ema, epochs=2)

    assert module.net.weight.item() != pytest.approx(INITIAL)


def test_the_averaged_weights_replace_the_live_ones_at_the_end() -> None:
    """The averaged model is what the run is for, so it is what training leaves behind."""
    averaged, _ = fit(EmaWeights(decay=0.9), epochs=2)
    plain, _ = fit(EmaWeights(decay=0.9, after=0.99), epochs=2)

    assert averaged.net.weight.item() != pytest.approx(plain.net.weight.item())


def test_a_weights_only_checkpoint_is_refused(tmp_path: Path) -> None:
    """Lightning skips callback hooks on that path, so the file would hold the wrong weights."""
    with pytest.raises(ValueError, match="save_weights_only"):
        fit(EmaWeights(decay=0.9), extra=[ModelCheckpoint(save_weights_only=True)], root=tmp_path)


def test_a_full_checkpoint_alongside_is_fine(tmp_path: Path) -> None:
    fit(EmaWeights(decay=0.9), extra=[ModelCheckpoint(save_weights_only=False)], root=tmp_path)


@pytest.mark.parametrize("decay", [0.0, 1.0, 1.5, -0.1])
def test_a_decay_outside_the_unit_interval_is_refused(decay: float) -> None:
    with pytest.raises(ValueError, match="decay"):
        EmaWeights(decay=decay)


@pytest.mark.parametrize("after", [1.0, 1.5, -0.1])
def test_a_warmup_that_covers_the_whole_run_is_refused(after: float) -> None:
    with pytest.raises(ValueError, match="after"):
        EmaWeights(after=after)


def test_it_is_reachable_from_config_by_name() -> None:
    assert isinstance(callback_registry.create("ema", decay=0.99), EmaWeights)


def saved(root: Path) -> dict[str, Any]:
    """The one checkpoint file the run wrote."""
    written = sorted(root.rglob("*.ckpt"))
    loaded: dict[str, Any] = torch.load(written[0], weights_only=False)
    return loaded


def test_a_weights_only_checkpoint_holds_the_averaged_weights(tmp_path: Path) -> None:
    """Lightning skips the save hook on this path, so the swap is what gets them in."""
    ema = EmaWeights(decay=0.9)

    module, _ = fit(ema, extra=[EmaModelCheckpoint(save_weights_only=True)], root=tmp_path)

    kept = saved(tmp_path)["state_dict"]["net.weight"].item()
    # The last thing validation saw was the average, and no step has moved it since. Without
    # the lending this would be the live weight instead, which this run drives elsewhere.
    assert kept == pytest.approx(module.validated[-1])
    assert "current_model_state" not in saved(tmp_path)  # so the stock hook really did not run


def test_the_live_weights_are_back_once_the_file_is_written() -> None:
    """A lent copy that is never returned would train on the average from then on."""
    ema = EmaWeights(decay=0.9)
    module = Recorder()
    fit(ema, epochs=1)
    before = ema.state_dict()

    with ema.averaged_weights(module):
        pass

    assert ema.state_dict() == before
    assert module.net.weight.item() == pytest.approx(INITIAL)


def test_a_full_checkpoint_takes_the_ordinary_path(tmp_path: Path) -> None:
    """Lightning's own hook already handles it, and it also stores the live weights."""
    fit(EmaWeights(decay=0.9), extra=[EmaModelCheckpoint(save_weights_only=False)], root=tmp_path)

    assert "current_model_state" in saved(tmp_path)


def test_a_run_without_averaging_saves_what_it_trained(tmp_path: Path) -> None:
    """There is no average to lend during warmup, so the live weights are the right ones."""
    module, _ = fit(
        EmaWeights(decay=0.9, after=0.99), extra=[EmaModelCheckpoint(save_weights_only=True)], root=tmp_path
    )

    assert saved(tmp_path)["state_dict"]["net.weight"].item() == pytest.approx(module.net.weight.item())


def test_it_is_an_ordinary_checkpoint_without_ema(tmp_path: Path) -> None:
    module = Recorder()
    data = DataLoader(TensorDataset(torch.ones(STEPS_PER_EPOCH, 1)), batch_size=1)
    trainer = quiet_trainer(
        default_root_dir=tmp_path,
        enable_checkpointing=True,
        callbacks=[EmaModelCheckpoint(save_weights_only=True)],
    )
    trainer.fit(module, data, data)

    assert saved(tmp_path)["state_dict"]["net.weight"].item() == pytest.approx(module.net.weight.item())


def test_the_refusal_points_at_the_checkpoint_that_works(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ema_checkpoint"):
        fit(EmaWeights(decay=0.9), extra=[ModelCheckpoint(save_weights_only=True)], root=tmp_path)


def test_both_are_reachable_from_config_by_name() -> None:
    assert isinstance(callback_registry.create("ema_checkpoint", save_weights_only=True), EmaModelCheckpoint)
