"""Moving one number of an objective over the run — a focal gamma, a label smoothing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import lightning as L
import pytest
import torch
from torch import Tensor, nn

from src.callbacks.anneal import RAMPS, scheduled
from src.core import LossOutput
from src.losses import Loss
from tests.unit.callbacks.conftest import prepared

HERE = "tests.unit.callbacks.test_anneal"


class Trail(L.Callback):
    """Every epoch's value of the annealed number, so a ramp can be read rather than only its end."""

    seen: list[float] = []  # noqa: RUF012 -- declared by `_target_`, so the run cannot hand it a list

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        module: Any = pl_module
        type(self).seen.append(float(module.learner.loss_of("species").module.label_smoothing))


class Learned(Loss):
    """A loss whose number is a parameter the optimizer owns, which is the one kind that must not move."""

    def __init__(self) -> None:
        super().__init__()
        self.margin = nn.Parameter(torch.tensor(0.5))

    def forward(self, outputs: Tensor, targets: Tensor) -> LossOutput:
        return self.reported((outputs.mean() - self.margin).abs())


def annealing(**declared: Any) -> list[dict[str, Any]]:
    return [{"name": "anneal", "task": "species", "parameter": "label_smoothing", "start": 0.0, "end": 0.2, **declared}]


def loss_of(declaration: Mapping[str, Any], **overrides: Any) -> Any:
    built = prepared(declaration, **overrides)
    built.trainer.fit(built.module, datamodule=built.data)
    return built.module.learner.loss_of("species")


@pytest.mark.parametrize(
    ("epoch", "value"),
    [
        pytest.param(0, 0.0, id="the first epoch is exactly the start"),
        pytest.param(2, 0.5, id="halfway is halfway, on a linear ramp"),
        pytest.param(4, 1.0, id="the window's end is exactly the end"),
        pytest.param(9, 1.0, id="and it holds there afterwards"),
    ],
)
def test_the_ramp_is_a_pure_function_of_the_epoch(epoch: int, value: float) -> None:
    """Pure in the epoch, which is what lets a resumed run pick the ramp up rather than start it over."""
    assert scheduled(epoch, window=5, start=0.0, end=1.0, shape=RAMPS["linear"]) == pytest.approx(value)


def test_a_window_of_one_epoch_is_a_step_rather_than_a_ramp() -> None:
    """The first epoch keeps the number it was constructed with; from the next one on, it is the end."""
    assert scheduled(0, window=1, start=0.0, end=1.0, shape=RAMPS["linear"]) == pytest.approx(0.0)
    assert scheduled(1, window=1, start=0.0, end=1.0, shape=RAMPS["linear"]) == pytest.approx(1.0)


def trail(declaration: Mapping[str, Any], **declared: Any) -> list[float]:
    """The annealed number as each epoch left it, over a four-epoch run."""
    Trail.seen = []
    built = prepared(declaration, epochs=4, callbacks=[*annealing(**declared), {"_target_": f"{HERE}.Trail"}])
    built.trainer.fit(built.module, datamodule=built.data)
    return Trail.seen


def test_the_ramp_is_spread_over_the_window_the_run_declared(declaration: Mapping[str, Any]) -> None:
    """`over` is what makes a run end on the objective it will be judged by rather than still moving it.

    Read across the epochs rather than at the end, because every window ends at `end`: what `over`
    changes is when it gets there.
    """
    whole = trail(declaration)
    half = trail(declaration, over=0.5)

    assert half[1] == pytest.approx(0.2), "half a four-epoch run is two, so the second one is already there"
    assert whole[1] < 0.2, "spread over all four, it is still on the way"


def test_the_number_reaches_the_term_that_holds_it(declaration: Mapping[str, Any]) -> None:
    """It sits on the library's own module, a level below the term that reports it, and is found there."""
    annealed = loss_of(declaration, epochs=2, callbacks=annealing())

    assert annealed.module.label_smoothing == pytest.approx(0.2)


def test_the_number_is_reported_under_a_family_of_its_own(declaration: Mapping[str, Any]) -> None:
    """A run that ends with a different objective than it started with must say so somewhere."""
    built = prepared(declaration, epochs=2, callbacks=annealing())

    built.trainer.fit(built.module, datamodule=built.data)

    assert "schedule/species/label_smoothing" in built.trainer.callback_metrics


def test_a_number_held_by_more_than_one_term_asks_which(declaration: Mapping[str, Any]) -> None:
    tasks = {
        "species": {
            **declaration["tasks"]["species"],
            "loss": [
                {"loss": "cross_entropy", "log_name": "soft", "weight": 0.5},
                {"loss": "cross_entropy", "log_name": "hard", "weight": 0.5},
            ],
        }
    }
    built = prepared(declaration, tasks=tasks, callbacks=annealing())

    with pytest.raises(ValueError, match=r"soft\.label_smoothing"):
        built.trainer.fit(built.module, datamodule=built.data)


def test_naming_the_term_resolves_it(declaration: Mapping[str, Any]) -> None:
    tasks = {
        "species": {
            **declaration["tasks"]["species"],
            "loss": [
                {"loss": "cross_entropy", "log_name": "soft", "weight": 0.5},
                {"loss": "cross_entropy", "log_name": "hard", "weight": 0.5},
            ],
        }
    }

    annealed = loss_of(declaration, tasks=tasks, epochs=2, callbacks=annealing(parameter="soft.label_smoothing"))

    soft, hard = annealed.parts
    assert soft.module.label_smoothing == pytest.approx(0.2)
    assert hard.module.label_smoothing == pytest.approx(0.0), "the term nobody named is left alone"


def test_a_number_the_optimizer_owns_is_refused(declaration: Mapping[str, Any]) -> None:
    """Writing over a parameter fights the optimizer: the ramp would undo every step it takes."""
    tasks = {
        "species": {
            **declaration["tasks"]["species"],
            "loss": {"_target_": "tests.unit.callbacks.test_anneal.Learned"},
        }
    }
    built = prepared(declaration, tasks=tasks, callbacks=annealing(parameter="margin"))

    with pytest.raises(ValueError, match="Parameter"):
        built.trainer.fit(built.module, datamodule=built.data)


def test_a_number_that_is_nowhere_names_what_is_there(declaration: Mapping[str, Any]) -> None:
    built = prepared(declaration, callbacks=annealing(parameter="temperature"))

    with pytest.raises(ValueError, match="label_smoothing"):
        built.trainer.fit(built.module, datamodule=built.data)


def test_a_ramp_shape_it_does_not_know_is_refused_where_it_was_declared(declaration: Mapping[str, Any]) -> None:
    with pytest.raises(ValueError, match="cosine"):
        prepared(declaration, callbacks=annealing(schedule="exponential"))
