"""Unit tests for training callbacks."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch

from src.callbacks.ema import EmaCallback
from src.callbacks.freeze import FreezeCallback
from src.callbacks.metric_summary import MetricSummaryCallback
from src.callbacks.model_summary import TreeModelSummary, tree_names
from src.callbacks.registry import callback_registry
from tests.support.fakes import FakePlotLogger, TinyLitModule, make_mock_trainer

# ---------------------------------------------------------------- EmaCallback: init


class TestEmaCallbackInit:
    def test_valid_params(self) -> None:
        cb = EmaCallback(decay=0.999, warmup_fraction=0.1, use_buffers=False)
        assert cb._warmup_fraction == 0.1
        assert cb._use_buffers is False

    @pytest.mark.parametrize(
        ("build", "match"),
        [
            # decay must be strictly inside (0, 1)
            pytest.param(lambda: EmaCallback(decay=0.0), "decay", id="decay-zero"),
            pytest.param(lambda: EmaCallback(decay=1.0), "decay", id="decay-one"),
            pytest.param(lambda: EmaCallback(decay=1.5), "decay", id="decay-above-one"),
            pytest.param(lambda: EmaCallback(decay=-0.1), "decay", id="decay-negative"),
            # warmup_fraction must be inside [0, 1)
            pytest.param(lambda: EmaCallback(warmup_fraction=-0.1), "warmup_fraction", id="warmup-negative"),
            pytest.param(lambda: EmaCallback(warmup_fraction=1.0), "warmup_fraction", id="warmup-one"),
            pytest.param(lambda: EmaCallback(warmup_fraction=2.0), "warmup_fraction", id="warmup-above-one"),
        ],
    )
    def test_invalid_constructor_argument_raises(self, build: Callable[[], EmaCallback], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            build()

    def test_warmup_fraction_zero_is_valid(self) -> None:
        EmaCallback(warmup_fraction=0.0)  # should not raise


# ---------------------------------------------------------------- EmaCallback: setup
#
# The EMA mechanics (averaging, validation swap, checkpoint persistence) are
# Lightning's ``EMAWeightAveraging`` and tested upstream. Here we only cover the
# thin facade: the averaged model is created on fit, and fractional warmup is
# resolved to Lightning's absolute ``update_starting_at_step``.


class TestEmaCallbackSetup:
    def test_average_model_created_on_fit(self) -> None:
        cb = EmaCallback(decay=0.999)
        cb.setup(make_mock_trainer(), TinyLitModule(), stage="fit")
        assert cb._average_model is not None

    def test_no_average_model_outside_fit(self) -> None:
        cb = EmaCallback(decay=0.999)
        cb.setup(make_mock_trainer(), TinyLitModule(), stage="validate")
        assert cb._average_model is None

    def test_warmup_fraction_resolves_to_start_step(self) -> None:
        cb = EmaCallback(decay=0.999, warmup_fraction=0.5)
        cb.setup(make_mock_trainer(estimated_stepping_batches=200), TinyLitModule(), stage="fit")
        assert cb.update_starting_at_step == 100

    def test_zero_warmup_starts_at_step_zero(self) -> None:
        cb = EmaCallback(decay=0.999, warmup_fraction=0.0)
        cb.setup(make_mock_trainer(estimated_stepping_batches=200), TinyLitModule(), stage="fit")
        assert cb.update_starting_at_step == 0


# ---------------------------------------------------------------- EmaCallback: warmup validation guard
#
# Until the first EMA update the averaged model is a frozen copy of the initial weights.
# Lightning swaps it in for every validation unconditionally, so validation during warmup
# would evaluate the untrained model. The guard skips the swap until averaging has started.


class TestEmaWarmupValidationGuard:
    @staticmethod
    def _prepared(latest_update_step: int) -> tuple[EmaCallback, TinyLitModule]:
        """Averaged model captures the init weights (ones); the live model is then diverged to twos."""
        module = TinyLitModule()  # linear.weight initialised to ones -> AveragedModel copies ones
        cb = EmaCallback(decay=0.999, warmup_fraction=0.5)
        cb.setup(make_mock_trainer(estimated_stepping_batches=200), module, stage="fit")
        with torch.no_grad():
            module.linear.weight.copy_(torch.full_like(module.linear.weight, 2.0))
        cb._latest_update_step = latest_update_step
        return cb, module

    def test_no_swap_during_warmup(self) -> None:
        """Before any EMA update, validation keeps the live (training) weights — no swap to init."""
        cb, module = self._prepared(latest_update_step=0)
        cb.on_validation_epoch_start(make_mock_trainer(), module)
        assert torch.allclose(module.linear.weight, torch.full_like(module.linear.weight, 2.0))

    def test_swap_after_averaging_started(self) -> None:
        """Once averaging has begun, the averaged weights (ones) are swapped in for validation."""
        cb, module = self._prepared(latest_update_step=5)
        cb.on_validation_epoch_start(make_mock_trainer(), module)
        assert torch.allclose(module.linear.weight, torch.ones_like(module.linear.weight))

    def test_validation_swap_is_symmetric_after_start(self) -> None:
        """start then end restores the live weights (averaged swapped in, then back out)."""
        cb, module = self._prepared(latest_update_step=5)
        cb.on_validation_epoch_start(make_mock_trainer(), module)
        cb.on_validation_epoch_end(make_mock_trainer(), module)
        assert torch.allclose(module.linear.weight, torch.full_like(module.linear.weight, 2.0))

    def test_train_end_keeps_trained_weights_if_averaging_never_ran(self) -> None:
        """The degenerate no-averaging case must not overwrite the trained model with init weights."""
        cb, module = self._prepared(latest_update_step=0)
        cb.on_train_end(make_mock_trainer(), module)
        assert torch.allclose(module.linear.weight, torch.full_like(module.linear.weight, 2.0))

    def test_checkpoint_during_warmup_keeps_live_weights(self) -> None:
        """A checkpoint saved before averaging starts must store the live (training) weights.

        Without the guard the parent replaces state_dict with the averaged model — during warmup
        that is the frozen init weights, while the monitored metric came from the live weights, so
        a 'best' checkpoint would silently hold random weights.
        """
        cb, module = self._prepared(latest_update_step=0)
        checkpoint = {"state_dict": {key: value.clone() for key, value in module.state_dict().items()}}
        cb.on_save_checkpoint(make_mock_trainer(), module, checkpoint)
        assert torch.allclose(checkpoint["state_dict"]["linear.weight"], torch.full((2, 2), 2.0))
        assert "current_model_state" not in checkpoint

    def test_checkpoint_after_averaging_started_stores_averaged_weights(self) -> None:
        """Once averaging runs, the parent behaviour applies: state_dict = EMA, live kept aside."""
        cb, module = self._prepared(latest_update_step=5)
        checkpoint = {"state_dict": {key: value.clone() for key, value in module.state_dict().items()}}
        cb.on_save_checkpoint(make_mock_trainer(), module, checkpoint)
        assert torch.allclose(checkpoint["state_dict"]["linear.weight"], torch.ones(2, 2))
        assert torch.allclose(checkpoint["current_model_state"]["linear.weight"], torch.full((2, 2), 2.0))


# --------------------------- EmaCallback: private-API canary (Lightning coupling)
#
# Our guard reads Lightning's private ``_latest_update_step``. The tests above SET that
# attribute themselves, so they would not notice if Lightning renamed it (assigning an
# unknown attribute is not an error in Python). This canary instead drives the parent's
# real ``on_train_batch_end`` and asserts the guard flips — so a rename on a version bump
# fails loudly here instead of silently validating init weights in production.


class TestEmaPrivateApiCanary:
    def test_real_update_path_flips_the_guard(self) -> None:
        module = TinyLitModule()
        cb = EmaCallback(decay=0.999, warmup_fraction=0.0)  # update_starting_at_step == 0
        trainer = make_mock_trainer(global_step=1, estimated_stepping_batches=100)
        cb.setup(trainer, module, stage="fit")
        assert not cb._averaging_has_started()  # nothing averaged yet

        cb.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=0)

        # The parent's real code set _latest_update_step via the exact name our guard reads.
        assert cb._averaging_has_started()


# ---------------------------------------------------------------- FreezeCallback: init


class TestFreezeCallbackInit:
    def test_valid_params(self) -> None:
        cb = FreezeCallback(targets=["model.backbone"])
        assert cb._targets == ["model.backbone"]

    def test_empty_targets_rejected(self) -> None:
        with pytest.raises(ValueError, match="targets"):
            FreezeCallback(targets=[])

    @pytest.mark.parametrize("bad_fraction", [0.0, 1.5, -0.1])
    def test_float_unfreeze_at_must_be_in_0_1(self, bad_fraction: float) -> None:
        with pytest.raises(ValueError, match="unfreeze_at"):
            FreezeCallback(targets=["model.backbone"], unfreeze_at=bad_fraction)

    def test_float_unfreeze_at_1_0_is_valid(self) -> None:
        FreezeCallback(targets=["model.backbone"], unfreeze_at=1.0)

    def test_minus_one_means_never_unfreeze(self) -> None:
        cb = FreezeCallback(targets=["model.backbone"], unfreeze_at=-1)
        assert cb._resolve_epoch(max_epochs=10) is None


# ---------------------------------------------------------------- TreeModelSummary


class TestTreeModelSummary:
    def test_tree_names_builds_connectors(self) -> None:
        names = [
            "model",
            "model.backbone",
            "model.backbone.encoder",
            "model.backbone.decoder",
            "model.heads",
            "model.heads.species",
            "model.heads.mask",
        ]
        assert tree_names(names) == [
            "model",
            "├─ backbone",
            "│  ├─ encoder",
            "│  └─ decoder",
            "└─ heads",
            "   ├─ species",
            "   └─ mask",
        ]

    def test_tree_names_single_root(self) -> None:
        assert tree_names(["model"]) == ["model"]

    def test_registered_under_model_summary(self) -> None:
        callback = callback_registry.create("model_summary", max_depth=3)
        assert isinstance(callback, TreeModelSummary)
        assert callback._max_depth == 3

    def test_summarize_renders_tree_and_footer(self) -> None:
        from rich import get_console

        summary_data = [
            (" ", ["0", "1", "2"]),  # index column — must be dropped
            ("Name", ["model", "model.backbone", "model.heads"]),
            ("Type", ["CompositeModel", "SmpBackbone", "ModuleDict"]),
            ("Params", ["14.3 M", "14.3 M", "19.9 K"]),
            ("Mode", ["train", "train", "train"]),
            ("FLOPs", ["0", "0", "0"]),
        ]
        with get_console().capture() as capture:
            TreeModelSummary.summarize(
                summary_data,
                total_parameters=14_300_000,
                trainable_parameters=14_300_000,
                model_size=57.392,
                total_training_modes={"train": 159, "eval": 0},
                total_flops=0,
            )
        output = capture.get()
        assert "Model summary" in output  # blue title above the table
        assert "├─ backbone" in output and "└─ heads" in output  # tree connectors in Name column
        assert "Trainable params" in output and "Modules in train mode" in output  # footer kept
        assert "Total FLOPs" in output


# ---------------------------------------------------------------- MetricSummaryCallback


def _summarymake_mock_trainer(logger: object, *, global_zero: bool = True) -> MagicMock:
    trainer = MagicMock()
    trainer.is_global_zero = global_zero
    trainer.logger = logger
    trainer.callback_metrics = {
        "species/f1/test": torch.tensor(0.75),
        "breed/f1/test/mean": torch.tensor(0.16),
        "breed/f1/test/Abyssinian": torch.tensor(0.22),
        "loss/test/total": torch.tensor(4.87),
    }
    return trainer


class TestMetricSummaryCallback:
    def test_registered(self) -> None:
        assert "metric_summary" in callback_registry

    def test_reports_headline_metrics_to_plot_logger(self) -> None:
        logger = FakePlotLogger()
        MetricSummaryCallback().on_test_end(_summarymake_mock_trainer(logger), MagicMock())
        # Names match the training table rows: stage and the "mean" leaf are stripped.
        assert set(logger.single_values) == {"species/f1", "breed/f1", "loss/total"}
        assert logger.single_values["species/f1"] == pytest.approx(0.75, abs=1e-4)

    def test_noop_without_plot_logger(self) -> None:
        """A logger lacking the single-value capability is skipped, not crashed."""
        cb = MetricSummaryCallback()
        cb.on_test_end(_summarymake_mock_trainer(object()), MagicMock())  # object() is not a PlotLogger

    def test_skips_non_global_zero(self) -> None:
        logger = FakePlotLogger()
        MetricSummaryCallback().on_test_end(_summarymake_mock_trainer(logger, global_zero=False), MagicMock())
        assert logger.single_values == {}
