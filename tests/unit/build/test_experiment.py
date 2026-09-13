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
from src.training import model_weights, restore_best_weights
from src.training.checkpoints import MODEL_PREFIX


def written(path: Path, weights: Mapping[str, Any]) -> str:
    """A checkpoint shaped as a run writes one: the whole module's state, the model inside it."""
    torch.save({"state_dict": {f"{MODEL_PREFIX}{name}": value for name, value in weights.items()}}, path)
    return str(path)


def experiment(declaration: Mapping[str, Any], **overrides: Any) -> Experiment:
    return build(load_config({**declaration, **overrides}))


class Learned(nn.Module):
    """A learner as a checkpoint names one: the network under ``model``, and whatever learned beside it.

    What ``restore_best_weights`` is handed, spelled out rather than assembled — a run's own learner
    carries a backbone and a loss per task, and none of that is what these tests are about.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Linear(2, 2)


class TestWeights:
    def test_a_checkpoint_of_ours_gives_up_the_models_own_keys(self, tmp_path: Path) -> None:
        path = written(tmp_path / "one.ckpt", {"weight": torch.zeros(2, 2)})

        assert set(model_weights(path)) == {"weight"}

    def test_a_file_that_is_not_a_checkpoint_of_ours_is_refused_by_name(self, tmp_path: Path) -> None:
        torch.save({"weight": torch.zeros(2)}, tmp_path / "backbone.pt")

        with pytest.raises(ValueError, match="state_dict"):
            model_weights(str(tmp_path / "backbone.pt"))

    def test_weights_that_do_not_fit_are_refused_rather_than_half_loaded(self, tmp_path: Path) -> None:
        """A model with a loaded encoder and a fresh head looks trained and is not."""
        path = written(tmp_path / "other.ckpt", {"weight": torch.zeros(3, 3), "bias": torch.zeros(3)})

        with pytest.raises(ValueError, match="does not fit"):
            restore_best_weights(_kept(path), Learned())

    def test_the_checkpoint_a_run_kept_is_what_it_ends_holding(self, tmp_path: Path) -> None:
        """Lightning reloads nothing when the module is passed explicitly, so the run would ship its last epoch."""
        learner = Learned()
        path = written(tmp_path / "best.ckpt", {"weight": torch.zeros(2, 2), "bias": torch.zeros(2)})

        restore_best_weights(_kept(path), learner)

        assert torch.equal(learner.model.weight, torch.zeros(2, 2))

    def test_a_run_that_kept_nothing_ends_holding_what_it_trained(self) -> None:
        learner = Learned()
        before = learner.model.weight.clone()

        restore_best_weights(_kept(""), learner)

        assert torch.equal(learner.model.weight, before)


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
        assert all(torch.equal(held[name], value) for name, value in model_weights(str(kept)).items())

    def test_a_run_ends_holding_everything_it_learned_at_that_epoch_not_the_network_alone(
        self, declaration: Mapping[str, Any], tmp_path: Path
    ) -> None:
        """A loss with parameters of its own is learned by the run, so the epoch it keeps is the whole of it.

        Restoring the network alone leaves an angular margin's prototypes at the last epoch while every
        weight beside them comes from another one — a model that never existed at any point of the run.
        """
        declared = {
            **declaration,
            "epochs": 2,
            # As above: the rate is multiplied by a thousand once the first epoch is over, so the epoch
            # this run keeps is certainly not the one it ends on.
            "scheduler": {"name": "step", "step_size": 1, "gamma": 1000},
            "tasks": {
                name: {**task, "loss": {"_target_": "tests.support.losses.LearnedMargin"}}
                for name, task in declaration["tasks"].items()
            },
            "callbacks": [
                {"name": "checkpoint", "monitor": "val/loss", "mode": "min", "dirpath": str(tmp_path / "kept")}
            ],
        }
        built = experiment(declared)

        run(built)

        kept = Path(str(getattr(built.trainer.checkpoint_callback, "best_model_path", "")))
        assert "epoch=0" in kept.name, "the run got worse, so the epoch it kept is not the one it ended on"
        saved = torch.load(kept, map_location="cpu", weights_only=True)["state_dict"]
        margin = saved["learner.losses.species.margin"]
        assert torch.any(margin != 0), "the margin moved, so holding the kept one is a claim about something"
        held = built.module.learner.state_dict()
        adrift = [one for one, value in saved.items() if not torch.equal(held[one.removeprefix("learner.")], value)]
        assert not adrift, f"the run ended holding something the epoch it kept did not: {adrift}"

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


class TestWhatARunShips:
    """A run ends holding weights; what turns those into something deployable happens here and nowhere else."""

    def test_a_run_writes_every_declared_format_and_the_record_that_describes_them(
        self, declaration: Mapping[str, Any], tmp_path: Path
    ) -> None:
        """The milestone in one assertion: trained here, and readable by something that is not this run."""
        built = experiment(declaration, export=[{"name": "onnx"}, {"name": "pt2"}])

        manifest = run(built)

        directory = Path(declaration["run"]["directory"])
        assert [one.artifact for one in manifest.artifacts] == ["model.onnx", "model.pt2"]
        assert (directory / "model.onnx").exists() and (directory / "model.pt2").exists()
        assert (directory / "model.json").exists()
        assert [one.name for one in manifest.outputs] == ["species"]

    def test_a_run_that_declared_no_format_writes_nothing_and_says_nothing(
        self, declaration: Mapping[str, Any]
    ) -> None:
        """`export: none` is the default, and a run under it should not pay for a decision it did not make."""
        manifest = run(experiment(declaration))

        assert manifest.artifacts == ()
        assert not list(Path(declaration["run"]["directory"]).glob("model.*"))

    def test_the_record_reaches_the_tracker_the_run_declared(self, declaration: Mapping[str, Any]) -> None:
        """Where the hyperparameters went: the same run described from both ends, read side by side."""
        tracker = {"name": "csv", "save_dir": declaration["run"]["directory"], "version": ""}
        built = experiment(declaration, export=[{"name": "onnx"}], tracker=tracker)

        run(built)

        kept = list(Path(cast(Any, built.trainer.logger).log_dir).glob("model.json"))
        assert len(kept) == 1

    def test_a_run_is_left_able_to_carry_on_after_shipping(self, declaration: Mapping[str, Any]) -> None:
        """Writing moves the graph to the processor and turns training off; both belong to the caller."""
        built = experiment(declaration, export=[{"name": "onnx"}])
        built.module.learner.model.train()

        run(built)

        assert built.module.learner.model.training


def _kept(path: str) -> Any:
    """A trainer as ``restore_best_weights`` reads one: the path its checkpoint callback settled on."""
    return cast(Any, SimpleNamespace(checkpoint_callback=SimpleNamespace(best_model_path=path)))
