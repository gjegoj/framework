"""CriterionScheduleCallback: schedule math, constructor guards, resolution, application."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import cast

import lightning as L
import pytest

from src.callbacks.criterion_schedule import _SCHEDULES, CriterionScheduleCallback, scheduled_value
from src.callbacks.registry import callback_registry
from src.core.entities import Task
from src.losses.classification import FocalLoss
from src.losses.registry import criteria
from tests.support.builders import make_task


def _make_focal_task(task_name: str = "species") -> Task:
    task = make_task("classification", task_name, num_classes=3)
    return dataclasses.replace(task, criterion=criteria.create("focal", gamma=1.0))


def _focal_loss_of(task: Task) -> FocalLoss:
    """Narrow the wrapped loss for mypy (nn.Module attribute access is a Tensor|Module union)."""
    wrapped = task.criterion._loss  # noqa: SLF001 — pinning the resolver contract
    assert isinstance(wrapped, FocalLoss)
    return wrapped


def _fake_trainer(max_epochs: int | None, current_epoch: int = 0) -> L.Trainer:
    return cast(L.Trainer, SimpleNamespace(max_epochs=max_epochs, current_epoch=current_epoch))


def _fake_module(*tasks: object) -> tuple[L.LightningModule, list[tuple[str, float]]]:
    logged: list[tuple[str, float]] = []
    module = SimpleNamespace(tasks=list(tasks), log=lambda name, value: logged.append((name, value)))
    return cast(L.LightningModule, module), logged


class TestScheduledValue:
    """Pure schedule math: epoch 0 → start, window end and beyond → exactly end."""

    @pytest.mark.parametrize(
        ("kind", "epoch", "window", "expected"),
        [
            pytest.param("linear", 0, 5, 2.0, id="linear-epoch0-start"),
            pytest.param("linear", 2, 5, 1.25, id="linear-midpoint"),
            pytest.param("linear", 4, 5, 0.5, id="linear-window-end"),
            pytest.param("linear", 7, 5, 0.5, id="linear-past-window-clamps"),
            pytest.param("cosine", 0, 5, 2.0, id="cosine-epoch0-start"),
            pytest.param("cosine", 2, 5, 1.25, id="cosine-midpoint-is-half"),
            pytest.param("cosine", 4, 5, 0.5, id="cosine-window-end"),
            pytest.param("linear", 0, 1, 2.0, id="degenerate-window-stays-start"),
        ],
    )
    def test_value(self, kind: str, epoch: int, window: int, expected: float) -> None:
        value = scheduled_value(epoch, window, start=2.0, end=0.5, shape=_SCHEDULES[kind])
        assert value == pytest.approx(expected)


class TestConstructorGuards:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            pytest.param({"schedule": "exponential"}, "schedule", id="unknown-schedule-kind"),
            pytest.param({"over": 0.0}, "over", id="over-zero"),
            pytest.param({"over": 1.5}, "over", id="over-above-one"),
            pytest.param({"start": "2.0"}, "number", id="start-not-a-number"),
            pytest.param({"end": None}, "number", id="end-not-a-number"),
        ],
    )
    def test_invalid_constructor_argument_raises(self, kwargs: dict[str, object], match: str) -> None:
        arguments: dict[str, object] = {"task": "species", "parameter": "gamma", "start": 2.0, "end": 0.5}
        arguments.update(kwargs)
        with pytest.raises(ValueError, match=match):
            CriterionScheduleCallback(**arguments)  # type: ignore[arg-type]

    def test_valid_construction(self) -> None:
        callback = CriterionScheduleCallback(task="species", parameter="gamma", start=2.0, end=0.5)
        assert callback is not None


class TestResolution:
    def test_wrapped_loss_attribute_resolves_and_applies(self) -> None:
        task = _make_focal_task()
        module, logged = _fake_module(task)
        callback = CriterionScheduleCallback(task="species", parameter="gamma", start=2.0, end=0.5)
        callback.on_fit_start(_fake_trainer(max_epochs=5), module)
        callback.on_train_epoch_start(_fake_trainer(max_epochs=5, current_epoch=2), module)
        assert _focal_loss_of(task).gamma == pytest.approx(1.25)
        assert logged == [("schedule/species/gamma", pytest.approx(1.25))]

    def test_direct_attribute_resolves(self) -> None:
        focal_loss = _focal_loss_of(_make_focal_task())  # the bare FocalLoss: gamma is a direct attribute
        module, _ = _fake_module(SimpleNamespace(name="species", criterion=focal_loss))
        callback = CriterionScheduleCallback(task="species", parameter="gamma", start=2.0, end=0.5)
        callback.on_fit_start(_fake_trainer(max_epochs=5), module)
        callback.on_train_epoch_start(_fake_trainer(max_epochs=5, current_epoch=4), module)
        assert focal_loss.gamma == pytest.approx(0.5)

    def test_over_fraction_reaches_end_early_and_holds(self) -> None:
        task = _make_focal_task()
        module, _ = _fake_module(task)
        callback = CriterionScheduleCallback(task="species", parameter="gamma", start=2.0, end=0.5, over=0.5)
        callback.on_fit_start(_fake_trainer(max_epochs=10), module)  # window = 5
        callback.on_train_epoch_start(_fake_trainer(10, current_epoch=4), module)
        assert _focal_loss_of(task).gamma == pytest.approx(0.5)
        callback.on_train_epoch_start(_fake_trainer(10, current_epoch=9), module)
        assert _focal_loss_of(task).gamma == pytest.approx(0.5)


class TestResolutionGuards:
    def test_unknown_task_raises_listing_names(self) -> None:
        module, _ = _fake_module(_make_focal_task("species"))
        callback = CriterionScheduleCallback(task="breed", parameter="gamma", start=2.0, end=0.5)
        with pytest.raises(ValueError, match="species"):
            callback.on_fit_start(_fake_trainer(max_epochs=5), module)

    def test_unknown_parameter_raises_listing_numeric_attributes(self) -> None:
        module, _ = _fake_module(_make_focal_task())
        callback = CriterionScheduleCallback(task="species", parameter="gama", start=2.0, end=0.5)
        with pytest.raises(ValueError, match="gamma"):
            callback.on_fit_start(_fake_trainer(max_epochs=5), module)

    def test_learnable_parameter_refused(self) -> None:
        task = SimpleNamespace(name="pair", criterion=criteria.create("info_nce"))
        module, _ = _fake_module(task)
        callback = CriterionScheduleCallback(task="pair", parameter="logit_scale", start=2.0, end=0.5)
        with pytest.raises(ValueError, match="optimizer"):
            callback.on_fit_start(_fake_trainer(max_epochs=5), module)

    def test_non_numeric_attribute_refused(self) -> None:
        module, _ = _fake_module(_make_focal_task())
        callback = CriterionScheduleCallback(task="species", parameter="reduction", start=2.0, end=0.5)
        with pytest.raises(ValueError, match="numeric"):
            callback.on_fit_start(_fake_trainer(max_epochs=5), module)

    def test_module_without_tasks_raises(self) -> None:
        callback = CriterionScheduleCallback(task="species", parameter="gamma", start=2.0, end=0.5)
        with pytest.raises(ValueError, match="tasks"):
            callback.on_fit_start(_fake_trainer(max_epochs=5), cast(L.LightningModule, SimpleNamespace()))


class TestNoOpWithoutMaxEpochs:
    def test_unknown_max_epochs_warns_and_leaves_value(self) -> None:
        task = _make_focal_task()
        module, logged = _fake_module(task)
        callback = CriterionScheduleCallback(task="species", parameter="gamma", start=2.0, end=0.5)
        callback.on_fit_start(_fake_trainer(max_epochs=None), module)
        callback.on_train_epoch_start(_fake_trainer(max_epochs=None, current_epoch=0), module)
        assert _focal_loss_of(task).gamma == pytest.approx(1.0)  # constructed value untouched
        assert logged == []


class TestRegistration:
    def test_registered(self) -> None:
        assert "criterion_schedule" in callback_registry
