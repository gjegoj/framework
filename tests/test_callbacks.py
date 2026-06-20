"""Unit tests for training callbacks."""

from __future__ import annotations

from unittest.mock import MagicMock

import lightning as L
import pytest
import torch
import torch.nn as nn

from src.callbacks.ema import EmaCallback
from src.callbacks.freeze import FreezeCallback
from src.callbacks.metric_summary import MetricSummaryCallback
from src.callbacks.model_summary import TreeModelSummary, tree_names
from src.callbacks.registry import callback_registry
from tests.test_metrics import FakePlotLogger

# ---------------------------------------------------------------- helpers


class _TinyModule(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        nn.init.ones_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)  # type: ignore[no-any-return]

    def training_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        return torch.tensor(0.0)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _trainer(global_step: int = 1, estimated_stepping_batches: int = 100) -> MagicMock:
    t = MagicMock()
    t.global_step = global_step
    t.estimated_stepping_batches = estimated_stepping_batches
    t.max_epochs = 10
    return t


# ---------------------------------------------------------------- EmaCallback: init


class TestEmaCallbackInit:
    def test_valid_params(self) -> None:
        cb = EmaCallback(decay=0.999, warmup_fraction=0.1, use_buffers=False)
        assert cb._warmup_fraction == 0.1
        assert cb._use_buffers is False

    def test_decay_must_be_strictly_between_0_and_1(self) -> None:
        for bad in (0.0, 1.0, 1.5, -0.1):
            with pytest.raises(ValueError, match="decay"):
                EmaCallback(decay=bad)

    def test_warmup_fraction_must_be_in_0_1_exclusive_right(self) -> None:
        for bad in (-0.1, 1.0, 2.0):
            with pytest.raises(ValueError, match="warmup_fraction"):
                EmaCallback(warmup_fraction=bad)

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
        cb.setup(_trainer(), _TinyModule(), stage="fit")
        assert cb._average_model is not None

    def test_no_average_model_outside_fit(self) -> None:
        cb = EmaCallback(decay=0.999)
        cb.setup(_trainer(), _TinyModule(), stage="validate")
        assert cb._average_model is None

    def test_warmup_fraction_resolves_to_start_step(self) -> None:
        cb = EmaCallback(decay=0.999, warmup_fraction=0.5)
        cb.setup(_trainer(estimated_stepping_batches=200), _TinyModule(), stage="fit")
        assert cb.update_starting_at_step == 100

    def test_zero_warmup_starts_at_step_zero(self) -> None:
        cb = EmaCallback(decay=0.999, warmup_fraction=0.0)
        cb.setup(_trainer(estimated_stepping_batches=200), _TinyModule(), stage="fit")
        assert cb.update_starting_at_step == 0


# ---------------------------------------------------------------- FreezeCallback: init


class TestFreezeCallbackInit:
    def test_valid_params(self) -> None:
        cb = FreezeCallback(targets=["model.backbone"])
        assert cb._targets == ["model.backbone"]

    def test_empty_targets_rejected(self) -> None:
        with pytest.raises(ValueError, match="targets"):
            FreezeCallback(targets=[])

    def test_float_unfreeze_at_must_be_in_0_1(self) -> None:
        for bad in (0.0, 1.5, -0.1):
            with pytest.raises(ValueError, match="unfreeze_at"):
                FreezeCallback(targets=["model.backbone"], unfreeze_at=bad)

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


def _summary_trainer(logger: object, *, global_zero: bool = True) -> MagicMock:
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
        MetricSummaryCallback().on_test_end(_summary_trainer(logger), MagicMock())
        # Names match the training table rows: stage and the "mean" leaf are stripped.
        assert set(logger.single_values) == {"species/f1", "breed/f1", "loss/total"}
        assert logger.single_values["species/f1"] == pytest.approx(0.75, abs=1e-4)

    def test_noop_without_plot_logger(self) -> None:
        """A logger lacking the single-value capability is skipped, not crashed."""
        cb = MetricSummaryCallback()
        cb.on_test_end(_summary_trainer(object()), MagicMock())  # object() is not a PlotLogger

    def test_skips_non_global_zero(self) -> None:
        logger = FakePlotLogger()
        MetricSummaryCallback().on_test_end(_summary_trainer(logger, global_zero=False), MagicMock())
        assert logger.single_values == {}
