"""The weights a run stops on: what test judges is what export ships."""

from __future__ import annotations

from pathlib import Path

import lightning as L
import pytest
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from src.assembly import assemble, run
from src.assembly.checkpoints import load_weights
from src.assembly.experiment import restore_best_weights
from tests.support.configs import disk_config
from tests.support.lightning import quiet_trainer


class Straight(L.LightningModule):
    """One linear layer under `model` — the shape a run's checkpoint actually has."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Linear(4, 2)

    def training_step(self, batch: tuple[Tensor, ...], batch_index: int) -> Tensor:
        inputs, targets = batch
        return nn.functional.mse_loss(self.model(inputs), targets)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


def loader() -> DataLoader[tuple[Tensor, ...]]:
    return DataLoader(TensorDataset(torch.randn(8, 4), torch.randn(8, 2)), batch_size=4)


def fitted(tmp_path: Path, checkpointing: bool) -> tuple[L.Trainer, Straight]:
    callbacks: list[Callback] = [ModelCheckpoint(dirpath=str(tmp_path / "kept"), save_top_k=1)] if checkpointing else []
    trainer = quiet_trainer(
        callbacks=callbacks,
        enable_checkpointing=checkpointing,
        default_root_dir=str(tmp_path),
    )
    module = Straight()
    trainer.fit(module, loader())
    return trainer, module


def test_the_kept_checkpoint_goes_back_into_the_module(tmp_path: Path) -> None:
    """Lightning does not do this: with the module passed explicitly it evaluates whatever is in memory."""
    trainer, module = fitted(tmp_path, checkpointing=True)
    kept = module.model.weight.detach().clone()
    with torch.no_grad():
        module.model.weight.add_(1.0)  # stand in for further training that the checkpoint did not keep

    restore_best_weights(trainer, module.model)

    assert torch.allclose(module.model.weight, kept)


def test_a_run_without_checkpointing_keeps_the_weights_it_has(tmp_path: Path) -> None:
    """A run that kept nothing has nothing to restore, and must not invent a path."""
    trainer, module = fitted(tmp_path, checkpointing=False)
    with torch.no_grad():
        module.model.weight.add_(1.0)
    after_training = module.model.weight.detach().clone()

    restore_best_weights(trainer, module.model)

    assert torch.allclose(module.model.weight, after_training)


def test_starting_weights_are_taken_without_the_training_state(tmp_path: Path) -> None:
    """Fine-tuning wants the weights and a fresh optimizer; only the state_dict travels."""
    trainer, source = fitted(tmp_path, checkpointing=True)
    kept = str(trainer.checkpoint_callback.best_model_path)  # type: ignore[union-attr]
    fresh = Straight()

    load_weights(fresh.model, kept)

    assert torch.allclose(fresh.model.weight, source.model.weight)


def test_a_run_that_started_from_a_checkpoint_does_not_end_holding_it(dataset_root: Path, tmp_path: Path) -> None:
    """Measured: handing the starting file to `test` reloads it over everything training produced.

    A run would then report the numbers of the weights it began with, and export
    those — which is why only `fit` is ever given a checkpoint.
    """
    settings = {"directory": str(tmp_path), "test": True}
    # A checkpoint Lightning itself wrote, so a reload would succeed rather than
    # trip over a missing key — the assertion has to be what fails.
    source_config = disk_config(dataset_root, run={**settings, "test": False})
    source = assemble(source_config)
    run(source, source_config)
    started_from = tmp_path / "start.ckpt"
    source.trainer.save_checkpoint(started_from)
    started = torch.load(started_from, map_location="cpu", weights_only=True)["state_dict"]

    config = disk_config(dataset_root, lr=0.1, run={**settings, "checkpoint_path": str(started_from)})
    experiment = assemble(config)
    run(experiment, config)

    ended = experiment.module.state_dict()
    moved = [
        key for key, value in started.items() if value.is_floating_point() and not torch.allclose(ended[key], value)
    ]
    assert moved, "the module ended holding the very weights it started from"


def test_a_plain_checkpoint_into_an_adapted_model_names_the_boundary(tmp_path: Path) -> None:
    """Measured: injection renames 50 of a ViT-tiny's keys, so the two checkpoints are not interchangeable.

    Renaming them into place would load a base weight beside an untrained delta
    and call it a warm start, so the refusal is the honest answer.
    """
    plain = tmp_path / "plain.ckpt"
    torch.save({"state_dict": Straight().state_dict()}, plain)

    class Renamed(L.LightningModule):
        """Weights one level deeper than the plain module's — what peft's rename looks like."""

        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Sequential(nn.Linear(4, 2))

    with pytest.raises(ValueError, match="adapters"):
        load_weights(Renamed().model, str(plain))


def test_the_run_folds_the_adapters_before_anything_reads_the_weights(dataset_root: Path, tmp_path: Path) -> None:
    """What `test` judges has to be what the artifact carries, and the fold is what makes them one."""
    config = disk_config(
        dataset_root,
        # img_size matches the shared transforms, which resize to 16x16; a ViT built for
        # 224 would refuse the batch before the fold could be observed.
        model={"name": "timm", "model_name": "vit_tiny_patch16_224", "pretrained": False, "img_size": 16},
        adapters={"name": "lora", "target_modules": ["fc1"], "rank": 4},
        run={"directory": str(tmp_path), "test": False},
    )
    experiment = assemble(config)
    assert any("lora_" in name for name in experiment.module.state_dict())

    run(experiment, config)

    assert not any("lora_" in name for name in experiment.module.state_dict())
