"""``MetricSummary``: the test stage's headline numbers reach the tracker's summary table."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import lightning as L
import pytest
import torch

from src.callbacks.metric_summary import MetricSummary, headline_metrics


def test_headline_keeps_scalars_and_vector_means_of_one_stage() -> None:
    """The summary shows aggregates: per-class leaves and other stages are noise here."""
    metrics = {
        "test/loss": torch.tensor(0.3),
        "test/label/f1": torch.tensor(0.8),
        "test/label/iou/mean": torch.tensor(0.5),
        "test/label/iou/cat": torch.tensor(0.4),
        "val/label/f1": torch.tensor(0.9),
    }

    headline = headline_metrics(metrics, "test")

    assert set(headline) == {"loss", "label/f1", "label/iou"}
    assert headline["label/iou"] == pytest.approx(0.5)


class RecordingSummary:
    """Structurally a ``SingleValueLogger`` — no inheritance, per the port canon."""

    def __init__(self) -> None:
        self.reported: dict[str, float] = {}

    def log_single_value(self, name: str, value: float) -> None:
        self.reported[name] = value


def _trainer(logger: object, is_global_zero: bool = True) -> L.Trainer:
    stub = SimpleNamespace(
        is_global_zero=is_global_zero,
        logger=logger,
        callback_metrics={"test/label/f1": torch.tensor(0.8)},
    )
    return cast("L.Trainer", stub)


def test_the_callback_reports_after_the_test_stage() -> None:
    backend = RecordingSummary()

    MetricSummary().on_test_end(_trainer(backend), cast("L.LightningModule", None))

    assert backend.reported["label/f1"] == pytest.approx(0.8)


def test_a_backend_without_a_summary_table_is_skipped_quietly() -> None:
    MetricSummary().on_test_end(_trainer(object()), cast("L.LightningModule", None))


def test_only_rank_zero_reports() -> None:
    backend = RecordingSummary()

    MetricSummary().on_test_end(_trainer(backend, is_global_zero=False), cast("L.LightningModule", None))

    assert backend.reported == {}
