"""An average of the weights, used in their place once there is one — and not a moment before."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor, nn

from src.callbacks.ema import EmaWeights
from src.experiment import Experiment
from src.training.checkpoints import MODEL_PREFIX
from tests.unit.callbacks.conftest import callback_of, prepared, weights_of


def fitted(declaration: Mapping[str, Any], **overrides: Any) -> torch.Tensor:
    """The weights a run ends with — every run here is seeded alike, so what differs is the callback."""
    built = prepared(declaration, **overrides)
    built.trainer.fit(built.module, datamodule=built.data)
    return weights_of(built.module.learner.model)


def averaging(**declared: Any) -> list[dict[str, Any]]:
    return [{"name": "ema", "decay": 0.5, **declared}]


def test_the_average_takes_the_place_of_the_weights_it_averaged(declaration: Mapping[str, Any]) -> None:
    assert not torch.equal(fitted(declaration, callbacks=averaging()), fitted(declaration))


def test_a_run_whose_averaging_never_begins_keeps_the_weights_it_trained(declaration: Mapping[str, Any]) -> None:
    """Lightning's copy holds the weights from *before* training until the first update, and three of
    its hooks use that copy regardless — so a warmup longer than the run would ship an untrained model."""
    assert torch.equal(fitted(declaration, callbacks=averaging(after=0.9)), fitted(declaration))


def test_averaging_begins_where_the_run_says_it_does(declaration: Mapping[str, Any]) -> None:
    built = prepared(declaration, epochs=2, callbacks=averaging(after=0.5))
    built.trainer.fit(built.module, datamodule=built.data)

    assert callback_of(built.trainer, EmaWeights).update_starting_at_step == 2, "half of a four-step run"


@pytest.mark.parametrize("decay", [0.0, 1.0, -0.5, 2.0], ids=repr)
def test_a_decay_that_is_not_a_share_of_the_average_is_refused(decay: float) -> None:
    with pytest.raises(ValueError, match="decay"):
        EmaWeights(decay=decay)


def test_saving_only_weights_beside_an_average_is_refused(declaration: Mapping[str, Any], tmp_path: Any) -> None:
    """Lightning runs a callback's save hook only for full checkpoints, so the file would hold the live
    weights while the metric it was chosen by came from the averaged ones."""
    saving = {"name": "checkpoint", "save_weights_only": True, "dirpath": str(tmp_path / "kept")}
    built = prepared(declaration, callbacks=[*averaging(), saving])

    with pytest.raises(ValueError, match="save_weights_only"):
        built.trainer.fit(built.module, datamodule=built.data)


def weights_in(state: Mapping[str, Tensor], model: nn.Module) -> Tensor:
    """The model's own weights out of a whole run's state, in the order the model keeps them."""
    return torch.cat([state[f"{MODEL_PREFIX}{name}"].flatten() for name, _ in model.named_parameters()])


def kept(declaration: Mapping[str, Any], directory: Path, **declared: Any) -> tuple[Experiment, dict[str, Any]]:
    """A run that both averages and saves, and the one file it decided to keep."""
    saving = {"name": "checkpoint", "monitor": "val/loss", "mode": "min", "dirpath": str(directory)}
    built = prepared(declaration, callbacks=[*averaging(**declared), saving])
    built.trainer.fit(built.module, datamodule=built.data)
    written = callback_of(built.trainer, ModelCheckpoint).best_model_path
    return built, torch.load(written, map_location="cpu", weights_only=False)


def test_the_file_a_run_keeps_holds_the_average_and_not_the_weights_it_averaged(
    declaration: Mapping[str, Any], tmp_path: Path
) -> None:
    """Measured on lightning 2.6.5: the average is written into the file's own ``state_dict`` and the
    live weights are moved aside to ``current_model_state``. So the file is about the weights its
    metric described, and everything that reads one back — restoring the best, shipping it — gets
    them without knowing an average was ever kept.
    """
    built, file = kept(declaration, tmp_path / "kept")
    model = built.module.learner.model

    assert torch.equal(weights_in(file["state_dict"], model), weights_of(model))
    assert not torch.equal(weights_in(file["current_model_state"], model), weights_of(model))


def test_a_file_kept_before_averaging_begins_holds_the_weights_it_was_measured_by(
    declaration: Mapping[str, Any], tmp_path: Path
) -> None:
    """Until the first update the copy holds the weights from before training, and Lightning writes it
    into the file regardless: a 'best' checkpoint would hold an untrained model under a metric that
    described a trained one."""
    built, file = kept(declaration, tmp_path / "kept", after=0.9)
    model = built.module.learner.model

    assert "current_model_state" not in file
    assert torch.equal(weights_in(file["state_dict"], model), weights_of(model))


def measured(declaration: Mapping[str, Any], **overrides: Any) -> Tensor:
    built = prepared(declaration, **overrides)
    built.trainer.fit(built.module, datamodule=built.data)
    return built.trainer.callback_metrics["val/loss"]


def test_validation_before_there_is_an_average_measures_the_live_weights(
    declaration: Mapping[str, Any],
) -> None:
    """Lightning swaps its copy in unconditionally, so the number reported for the first epochs would
    describe a model that has not been trained — and the saver would choose an epoch by it."""
    assert measured(declaration, callbacks=averaging(after=0.9)) == measured(declaration)
