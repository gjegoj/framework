"""Running a batch transform over training batches, for as long as the run wants it."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from src.callbacks.batch_transform import ApplyBatchTransform
from src.core import Batch
from src.tasks import Task
from src.transforms import MixUp
from tests.unit.callbacks.conftest import prepared

TRAIN_BATCHES = 2
"""Four training rows at two per batch — what the smallest run's split comes to."""


class Records:
    """A batch transform that rewrites nothing and remembers every batch it was asked about."""

    def __init__(self) -> None:
        self.seen = 0
        self.tasks: tuple[str, ...] = ()

    def for_tasks(self, tasks: Sequence[Task]) -> Records:
        self.tasks = tuple(task.name for task in tasks)
        return self

    def __call__(self, batch: Batch) -> Batch:
        self.seen += 1
        return batch


class Refuses(Records):
    """A transform that cannot serve this run's tasks, to see where that is said and how early."""

    def for_tasks(self, tasks: Sequence[Task]) -> Refuses:
        raise ValueError(f"cannot serve {', '.join(task.name for task in tasks)}")


def applying(transform: object, **declared: Any) -> list[dict[str, Any]]:
    return [{"_target_": "src.callbacks.batch_transform.ApplyBatchTransform", "transform": transform, **declared}]


def fitted(declaration: Mapping[str, Any], **overrides: Any) -> None:
    built = prepared(declaration, **overrides)
    built.trainer.fit(built.module, datamodule=built.data)


def test_it_runs_on_every_training_batch_and_on_nothing_else(declaration: Mapping[str, Any]) -> None:
    """One count carries both halves: the fit also validates, so a seam that did not ask which stage
    it was in would be counted above the training batches rather than equal to them. Evaluation is
    left alone by construction — the module reads the transform only while training — and this is
    what holds that construction in place."""
    recorded = Records()

    fitted(declaration, callbacks=applying(recorded))

    assert recorded.seen == TRAIN_BATCHES, "a batch outside training was rewritten too"


def test_it_is_bound_to_the_run_s_tasks_before_the_first_batch(declaration: Mapping[str, Any]) -> None:
    """Mixing rewrites every task's target, so a task it cannot serve is refused at setup, by name."""
    recorded = Records()

    fitted(declaration, callbacks=applying(recorded))

    assert recorded.tasks == ("species",)


def test_it_stops_at_the_epoch_the_run_declared(declaration: Mapping[str, Any]) -> None:
    """The last epochs are better left clean, so a run finishes on the data it is judged on."""
    recorded = Records()

    fitted(declaration, epochs=2, callbacks=applying(recorded, until=0.5))

    assert recorded.seen == TRAIN_BATCHES, "it kept going past the epoch it was declared until"


def test_a_task_the_transform_cannot_serve_is_refused_before_the_run_starts(
    declaration: Mapping[str, Any],
) -> None:
    """Which tasks a transform can serve is its own to say; this is where it gets asked, and early."""
    refusing = Refuses()

    with pytest.raises(ValueError, match="species"):
        fitted(declaration, callbacks=applying(refusing))

    assert refusing.seen == 0, "a batch was rewritten by something that had already refused the run"


def test_something_that_is_not_a_batch_transform_is_refused_where_it_was_declared() -> None:
    with pytest.raises(TypeError, match="Counter"):
        ApplyBatchTransform(transform=__import__("collections").Counter())


def test_a_second_one_is_refused_rather_than_left_to_overwrite_the_first(declaration: Mapping[str, Any]) -> None:
    """The module keeps one rewriting seam, so two of these do not compose — they race.

    Last to install wins, and the first to reach its `until` clears whatever is installed, the other
    one's included. Which of the two ran, and until when, would then follow from the order Lightning
    happens to call them in. Refused at setup, before a batch is read.
    """
    two = [*applying(Records()), *applying(Records(), until=0.5)]

    with pytest.raises(ValueError, match="one"):
        fitted(declaration, callbacks=two)


@pytest.mark.parametrize(
    ("kind", "declared"),
    [
        pytest.param(
            "classification",
            {"head": {"name": "cosine", "stream": "pooled"}, "loss": {"name": "arcface"}},
            id="prototypes in the network",
        ),
        pytest.param(
            {"name": "metric_learning", "embedding_dim": 8},
            # No vocabulary: this kind reads its column with the encoder that learns the identities of
            # the training split, and a declared one is refused where it is written.
            {"classes": None},
            id="prototypes in the objective",
        ),
    ],
)
def test_an_objective_that_cannot_read_a_blended_target_is_refused_before_the_first_batch(
    declaration: Mapping[str, Any], kind: Any, declared: Mapping[str, Any]
) -> None:
    """An angular margin is added to the one identity a sample is, and a blend leaves a share of two.

    Asked of the objective rather than read off the task's shape: the very same task under an ordinary
    cross-entropy blends perfectly well, so the shape of what it answers with does not settle this. Both
    arrangements are refused, because where the prototypes live does not change what the margin needs.
    """
    tasks = dict(declaration["tasks"])
    # A `None` in the parametrisation takes a key *out* of the declaration, which is how the kind that
    # learns its own vocabulary says it declares none.
    written = {**tasks["species"], "kind": kind, **declared}
    tasks["species"] = {key: value for key, value in written.items() if value is not None}
    mixing = MixUp()

    with pytest.raises(ValueError, match="species, learned by arcface"):
        fitted(declaration, tasks=tasks, callbacks=applying(mixing))
