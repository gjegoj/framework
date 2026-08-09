"""A step hands back what it produced; Lightning routes it to the batch-end hooks."""

from __future__ import annotations

from functools import partial
from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader

from src.core import Batch, Objective, Task, Topology
from src.core.entities import StepPreview, preview_of
from src.losses import CrossEntropyCriterion
from src.models import CompositeModel, LinearHead, TaskComponents
from src.tasks.adapters import as_class_indices
from src.training import TrainingModule
from tests.support.fakes import Batches, FlattenBackbone
from tests.support.lightning import quiet_trainer
from tests.support.narrowing import tensor


def module() -> TrainingModule:
    task = Task(
        name="label",
        topology=Topology.GLOBAL,
        objective=Objective.MULTICLASS,
        metrics={},
        class_names=["cat", "dog"],
    )
    return TrainingModule(
        model=CompositeModel(
            backbone=FlattenBackbone(dim=12),
            components={
                "label": TaskComponents(
                    head=LinearHead(12, 2),
                    criterion=CrossEntropyCriterion(),
                    activation=lambda logits: torch.softmax(logits, dim=1),
                    target_adapter=as_class_indices,
                )
            },
        ),
        tasks=[task],
        optimizer_factory=partial(torch.optim.SGD, lr=0.1),
    )


def batch() -> Batch:
    return Batch(inputs={"image": torch.rand(4, 3, 2, 2)}, targets={"label": torch.tensor([0, 1, 0, 1])})


class Reader(L.Callback):
    """A `StepPreviewConsumer`: it asks before the step, and reads where Lightning delivers.

    `wanted` is which batch indices it declares an interest in; `None` means all.
    """

    def __init__(self, wanted: set[int] | None = None) -> None:
        super().__init__()
        self.seen: list[tuple[int, tuple[int, ...]]] = []
        self.without: list[int] = []
        self._wanted = wanted
        self._awaiting = False

    @property
    def awaiting_preview(self) -> bool:
        return self._awaiting

    def on_validation_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._awaiting = self._wanted is None or batch_idx in self._wanted

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        preview = preview_of(outputs)
        if preview is None:
            self.without.append(batch_idx)
            return
        self.seen.append((batch_idx, tuple(tensor(preview.outputs["label"]).shape)))


class Bystander(L.Callback):
    """A callback that is not a consumer at all — the ordinary case, and the one that must cost nothing."""

    def __init__(self) -> None:
        super().__init__()
        self.saw_preview: list[bool] = []

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.saw_preview.append(preview_of(outputs) is not None)


def validated(*callbacks: L.Callback, batches: int = 3) -> None:
    loader = DataLoader(Batches([batch() for _ in range(batches)]), batch_size=None)
    trainer = quiet_trainer(callbacks=list(callbacks))
    trainer.validate(module(), dataloaders=loader, verbose=False)


def test_a_step_returns_its_loss_and_a_preview_of_what_it_produced() -> None:
    """Returned rather than remembered: no module state to keep, invalidate or let go stale."""
    result = module().validation_step(batch(), 0)

    assert set(result) == {"loss", StepPreview.KEY}
    preview = preview_of(result)
    assert preview is not None
    assert tensor(preview.outputs["label"]).shape == (4, 2)
    assert torch.equal(tensor(preview.targets["label"]), torch.tensor([0, 1, 0, 1]))


def test_a_preview_holds_no_graph_so_a_training_step_frees_what_it_built() -> None:
    """Returning the StepResult would carry the loss's grad_fn, the outputs' own, and the features."""
    preview = preview_of(module().training_step(batch(), 0))

    assert preview is not None
    assert not tensor(preview.outputs["label"]).requires_grad
    assert tensor(preview.outputs["label"]).grad_fn is None


def test_the_loss_still_reaches_the_loop_that_backpropagates_it() -> None:
    """The extra key must not cost Lightning the one thing it needs from a training step."""
    result = module().training_step(batch(), 0)

    assert result["loss"].requires_grad
    result["loss"].backward()


def test_a_consumer_reading_the_hook_argument_reads_its_own_batch() -> None:
    """Lightning passes a step's return value to every batch-end hook, per batch.

    That is the whole contract on the delivery side: no staleness, nothing kept.
    """
    validated(reader := Reader())

    assert reader.seen == [(0, (4, 2)), (1, (4, 2)), (2, (4, 2))]


def test_a_step_no_consumer_asked_about_builds_no_preview_at_all() -> None:
    """A preview shares storage with the activated outputs, and Lightning holds the
    step's return value through the optimizer step — so an unbuilt one is the whole
    saving. Measured at 352 MB for a `[16, 21, 512, 512]` batch, paid on every step
    of every run that never draws a page.
    """
    validated(bystander := Bystander())

    assert bystander.saw_preview == [False, False, False]


def test_a_consumer_is_handed_exactly_the_batches_it_asked_for() -> None:
    """The grid draws one batch every few epochs, so per-step is the granularity that matters.

    A flag resolved once per run would still pay on every step of the run that
    enabled it — which is the run that has the pictures big enough to matter.
    """
    validated(reader := Reader(wanted={1}))

    assert [index for index, _ in reader.seen] == [1]
    assert reader.without == [0, 2]


def test_a_module_called_outside_a_trainer_is_handed_everything() -> None:
    """There is nobody to ask and nothing to save, so withholding would only surprise."""
    result = module().validation_step(batch(), 0)

    assert preview_of(result) is not None


def test_anything_that_is_not_a_preview_reads_as_none() -> None:
    """A module of someone else's making returns what it likes; reading it must not raise."""
    assert preview_of(None) is None
    assert preview_of(torch.tensor(1.0)) is None
    assert preview_of({"loss": torch.tensor(1.0)}) is None
