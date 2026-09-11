"""What a run does with what was built: fit, evaluate, and end holding the weights it kept."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn

from src.build import build
from src.config import load_config
from src.experiment import Experiment, run
from src.training import restore_best_weights, shipped_weights
from src.training.checkpoints import MODEL_PREFIX


def written(path: Path, weights: Mapping[str, Any]) -> str:
    """A checkpoint shaped as a run writes one: the whole module's state, the model inside it."""
    torch.save({"state_dict": {f"{MODEL_PREFIX}{name}": value for name, value in weights.items()}}, path)
    return str(path)


def experiment(declaration: Mapping[str, Any], **overrides: Any) -> Experiment:
    return build(load_config({**declaration, **overrides}))


class TestWeights:
    def test_a_checkpoint_of_ours_gives_up_the_models_own_keys(self, tmp_path: Path) -> None:
        path = written(tmp_path / "one.ckpt", {"weight": torch.zeros(2, 2)})

        assert set(shipped_weights(path)) == {"weight"}

    def test_a_file_that_is_not_a_checkpoint_of_ours_is_refused_by_name(self, tmp_path: Path) -> None:
        torch.save({"weight": torch.zeros(2)}, tmp_path / "backbone.pt")

        with pytest.raises(ValueError, match="state_dict"):
            shipped_weights(str(tmp_path / "backbone.pt"))

    def test_weights_that_do_not_fit_are_refused_rather_than_half_loaded(self, tmp_path: Path) -> None:
        """A model with a loaded encoder and a fresh head looks trained and is not."""
        path = written(tmp_path / "other.ckpt", {"weight": torch.zeros(3, 3), "bias": torch.zeros(3)})
        model = nn.Linear(2, 2)

        with pytest.raises(ValueError, match="does not fit"):
            restore_best_weights(_kept(path), model)

    def test_the_checkpoint_a_run_kept_is_what_it_ends_holding(self, tmp_path: Path) -> None:
        """Lightning reloads nothing when the module is passed explicitly, so the run would ship its last epoch."""
        model = nn.Linear(2, 2)
        path = written(tmp_path / "best.ckpt", {"weight": torch.zeros(2, 2), "bias": torch.zeros(2)})

        restore_best_weights(_kept(path), model)

        assert torch.equal(model.weight, torch.zeros(2, 2))

    def test_a_run_that_kept_nothing_ends_holding_what_it_trained(self) -> None:
        model = nn.Linear(2, 2)
        before = model.weight.clone()

        restore_best_weights(_kept(""), model)

        assert torch.equal(model.weight, before)


class TestRun:
    def test_a_run_that_only_evaluates_never_touches_the_weights(self, declaration: Mapping[str, Any]) -> None:
        built = experiment(
            declaration,
            run={**declaration["run"], "train": False},
            trainer={**declaration["trainer"], "limit_test_batches": 1},
        )
        before = next(built.module.learner.model.parameters()).clone()

        run(built)

        assert torch.equal(next(built.module.learner.model.parameters()), before)

    def test_the_declaration_reaches_the_tracker_before_anything_can_fail(
        self, declaration: Mapping[str, Any], tmp_path: Path
    ) -> None:
        """A tracker should show what a run *is* whether or not the run turns out to finish."""
        declared = {
            **declaration,
            "tracker": {"name": "csv", "save_dir": str(tmp_path / "record")},
            "run": {**declaration["run"], "train": False, "test": False},
        }
        built = experiment(declared)

        run(built)

        assert Path(next((tmp_path / "record").rglob("hparams.yaml"))).read_text().count("resnet18") == 1

    def test_a_run_ends_holding_the_weights_it_kept_rather_than_its_last_epoch(
        self, declaration: Mapping[str, Any], tmp_path: Path
    ) -> None:
        """Lightning reloads nothing, so without this a monitored run reports one epoch and ships another."""
        declared = {
            **declaration,
            "epochs": 2,
            # The rate is multiplied by a thousand once the first epoch is over, so the second is
            # certainly the worse one — which makes the epoch this run keeps not the one it ends on.
            "scheduler": {"name": "step", "step_size": 1, "gamma": 1000},
            "callbacks": [
                {"name": "checkpoint", "monitor": "val/loss", "mode": "min", "dirpath": str(tmp_path / "kept")}
            ],
        }
        built = experiment(declared)

        run(built)

        kept = Path(str(getattr(built.trainer.checkpoint_callback, "best_model_path", "")))
        assert "epoch=0" in kept.name, "the run got worse, so the epoch it kept is not the one it ended on"
        held = built.module.learner.model.state_dict()
        assert all(torch.equal(held[name], value) for name, value in shipped_weights(str(kept)).items())

    def test_a_run_continues_from_a_checkpoint_the_shipped_saver_wrote(
        self, declaration: Mapping[str, Any], tmp_path: Path
    ) -> None:
        """The advertised workflow, end to end: `callbacks=default` writes it, `run.resume_path` reads it.

        A saver that kept weights alone would write a file Lightning refuses here — the optimizer and
        the epoch it needs to continue are exactly what such a file leaves out.
        """
        saver = {"name": "checkpoint", "monitor": "val/loss", "dirpath": str(tmp_path / "kept")}
        first = experiment(declaration, callbacks=[saver])
        run(first)
        kept = str(getattr(first.trainer.checkpoint_callback, "best_model_path", ""))

        second = experiment(declaration, epochs=2, callbacks=[saver], run={**declaration["run"], "resume_path": kept})
        run(second)

        assert second.trainer.current_epoch == 2, "it continued from the epoch the file was written at"

    def test_a_run_starts_from_the_weights_it_was_pointed_at(
        self, declaration: Mapping[str, Any], tmp_path: Path
    ) -> None:
        shaped = experiment(declaration)
        weights = {name: torch.zeros_like(value) for name, value in shaped.module.learner.model.state_dict().items()}
        built = experiment(
            declaration,
            run={
                **declaration["run"],
                "train": False,
                "test": False,
                "checkpoint_path": written(tmp_path / "w.ckpt", weights),
            },
        )

        run(built)

        assert all(torch.equal(value, torch.zeros_like(value)) for value in built.module.learner.model.parameters())


def _kept(path: str) -> Any:
    """A trainer as ``restore_best_weights`` reads one: the path its checkpoint callback settled on."""
    return cast(Any, SimpleNamespace(checkpoint_callback=SimpleNamespace(best_model_path=path)))
