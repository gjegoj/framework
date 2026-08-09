"""Callbacks are declared as a list of components, and reach the trainer in that order."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint

from src.assembly.callbacks import build_callbacks
from src.assembly.training import build_trainer
from tests.support.configs import paper_config


def test_no_section_means_no_callbacks() -> None:
    assert build_callbacks(paper_config()) == []


def test_each_entry_becomes_the_callback_it_names() -> None:
    built = build_callbacks(
        paper_config(
            callbacks=[
                {"name": "lr_monitor", "logging_interval": "epoch"},
                {"name": "checkpoint", "monitor": "val/loss", "mode": "min"},
            ]
        )
    )

    assert [type(callback) for callback in built] == [LearningRateMonitor, ModelCheckpoint]


def test_the_file_decides_the_order() -> None:
    """It is not cosmetic: a callback that changes the weights belongs before one that saves them."""
    built = build_callbacks(paper_config(callbacks=[{"name": "checkpoint", "monitor": "m"}, {"name": "lr_monitor"}]))

    assert [type(callback) for callback in built] == [ModelCheckpoint, LearningRateMonitor]


def test_two_checkpoints_on_different_metrics_are_two_entries() -> None:
    built = build_callbacks(
        paper_config(
            callbacks=[
                {"name": "checkpoint", "monitor": "val/loss", "mode": "min"},
                {"name": "checkpoint", "monitor": "val/label/accuracy", "mode": "max"},
            ]
        )
    )

    monitors = [callback.monitor for callback in built if isinstance(callback, ModelCheckpoint)]
    assert monitors == ["val/loss", "val/label/accuracy"]


def test_a_callback_of_your_own_needs_no_registration() -> None:
    built = build_callbacks(paper_config(callbacks=[{"_target_": "lightning.pytorch.callbacks.Timer"}]))

    assert len(built) == 1


def test_the_trainer_is_given_them() -> None:
    trainer = build_trainer(paper_config(callbacks=[{"name": "lr_monitor"}]))

    # Trainer fills `callbacks` in __init__, so its class body never declares it.
    registered: list[Callback] = trainer.callbacks  # type: ignore[attr-defined]
    assert any(isinstance(callback, LearningRateMonitor) for callback in registered)


def test_a_declared_logger_reaches_the_trainer(monkeypatch: Any) -> None:
    import sys
    from types import SimpleNamespace

    class _Task:
        name, id = "run", "abc"

        @classmethod
        def init(cls, **kwargs: Any) -> Any:
            return cls()

        def get_logger(self) -> Any:
            return SimpleNamespace()

    monkeypatch.setitem(sys.modules, "clearml", SimpleNamespace(Task=_Task))

    trainer = build_trainer(paper_config(logger={"name": "clearml", "project_name": "pets"}))

    assert type(trainer.logger).__name__ == "ClearMLLogger"


def test_no_logger_section_keeps_lightnings_default() -> None:
    trainer = build_trainer(paper_config())

    # Lightning's own default stands in — the framework adds no sentinel of its own.
    assert trainer.logger is not None


def test_a_declared_dirpath_reaches_the_saver() -> None:
    """Where a run's weights go is written in config, as `${run.directory}/checkpoints`.

    Nothing in assembly decides it, so this is the whole path from declaration to
    callback — and the shipped groups that carry that interpolation are checked in
    `test_run_directory.py`, which is where the guarantee now lives.
    """
    declared = paper_config(callbacks=[{"name": "checkpoint", "monitor": "val/loss", "dirpath": "runs/pets/one"}])

    trainer = build_trainer(declared)

    (saving,) = [one for one in trainer.callbacks if isinstance(one, ModelCheckpoint)]  # type: ignore[attr-defined]
    # Lightning realpaths what its constructor was given, so the assertion is on
    # the resolved form rather than on the string config wrote.
    assert saving.dirpath == str(Path("runs/pets/one").resolve())


def test_a_saver_without_a_dirpath_is_left_to_lightning() -> None:
    """Nothing is imposed here, so an experiment that says nothing gets Lightning's own
    answer — which is why every shipped group says something.
    """
    trainer = build_trainer(paper_config(callbacks=[{"name": "checkpoint", "monitor": "val/loss"}]))

    (saving,) = [one for one in trainer.callbacks if isinstance(one, ModelCheckpoint)]  # type: ignore[attr-defined]
    assert saving.dirpath is None
