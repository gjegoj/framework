"""``Freeze``: a backbone held still while the head learns, and let go on schedule."""

from __future__ import annotations

import logging
from functools import partial
from typing import Any, cast

import lightning as L
import pytest
import torch
from torch import nn

from src.callbacks import Freeze
from src.callbacks.registry import callback_registry
from src.losses import CrossEntropyCriterion
from src.models import CompositeModel, LinearHead, TaskComponents
from src.tasks.activations import softmax_probabilities
from src.tasks.adapters import as_class_indices
from src.training import TrainingModule
from tests.support.fakes import FlattenBackbone
from tests.support.lightning import quiet_trainer


class BackboneAndHead(nn.Module):
    """A stand-in module with the two parts a transfer-learning run has."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)


def module(inner: nn.Module) -> L.LightningModule:
    """The callback takes a LightningModule; only attribute lookup is exercised here."""
    return cast("L.LightningModule", inner)


def frozen(part: nn.Module) -> bool:
    return not any(parameter.requires_grad for parameter in part.parameters())


class _Trainable(L.LightningModule):
    """Something to freeze, something to keep learning, and a step for each stage."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Linear(4, 4)  # the part held still
        self.head = nn.Linear(4, 2)  # so the step still has a gradient to take

    def training_step(self, batch: Any, batch_index: int) -> torch.Tensor:
        inputs, targets = batch
        return nn.functional.cross_entropy(self.head(self.model(inputs)), targets)

    def test_step(self, batch: Any, batch_index: int) -> None:
        return None

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_the_named_module_is_frozen_before_training() -> None:
    parts = BackboneAndHead()

    Freeze(modules=["backbone"]).freeze_before_training(module(parts))

    assert frozen(parts.backbone)
    assert not frozen(parts.head)


def test_a_path_reaches_into_nested_modules() -> None:
    root = nn.Module()
    root.add_module("model", BackboneAndHead())

    Freeze(modules=["model.backbone"]).freeze_before_training(module(root))

    assert frozen(cast("Any", root).model.backbone)


def test_an_empty_target_list_is_refused() -> None:
    """Freezing nothing is a config mistake, not a no-op worth running."""
    with pytest.raises(ValueError, match="at least one"):
        Freeze(modules=[])


def test_a_path_that_names_nothing_lists_what_there_was() -> None:
    with pytest.raises(LookupError, match="backbone"):
        Freeze(modules=["encoder"]).freeze_before_training(module(BackboneAndHead()))


@pytest.mark.parametrize("until", [0.0, 1.5, -0.5])
def test_a_schedule_outside_the_run_is_refused(until: float) -> None:
    with pytest.raises(ValueError, match="until"):
        Freeze(modules=["backbone"], until=until)


def test_a_whole_number_is_an_epoch_and_a_fraction_is_a_share() -> None:
    assert Freeze(modules=["backbone"], until=3).release_epoch(max_epochs=10) == 3
    assert Freeze(modules=["backbone"], until=0.3).release_epoch(max_epochs=10) == 3


def test_a_share_follows_a_change_of_epoch_count() -> None:
    """That is the point of expressing it as a share rather than an epoch."""
    held = Freeze(modules=["backbone"], until=0.5)

    assert (held.release_epoch(max_epochs=10), held.release_epoch(max_epochs=100)) == (5, 50)


def test_never_unfreezing_is_the_default() -> None:
    assert Freeze(modules=["backbone"]).release_epoch(max_epochs=10) == 10  # an epoch the run never reaches


def test_it_is_reachable_from_config_by_name() -> None:
    assert isinstance(callback_registry.create("freeze", modules=["backbone"]), Freeze)


def test_the_documented_path_resolves_against_a_real_training_module() -> None:
    """The other tests build a stand-in named ``model``, so they cannot see this.

    ``targets: [model.backbone]`` is what the guide and this callback's own
    docstring tell a user to write, and it only works if the training module
    really exposes its model under that name.
    """
    backbone = FlattenBackbone(dim=4)
    built = TrainingModule(
        model=CompositeModel(
            backbone=backbone,
            components={
                "label": TaskComponents(
                    head=LinearHead(4, 2),
                    criterion=CrossEntropyCriterion(),
                    activation=softmax_probabilities,
                    target_adapter=as_class_indices,
                )
            },
        ),
        tasks=[],
        optimizer_factory=partial(torch.optim.SGD, lr=0.1),
    )

    Freeze(modules=["model.backbone"]).freeze_before_training(built)

    assert frozen(backbone)


def test_a_share_smaller_than_one_epoch_still_freezes_the_first() -> None:
    """int() would release at epoch 0 and the declared freeze would be a lie."""
    assert Freeze(modules=["backbone"], until=0.1).release_epoch(max_epochs=5) == 1


def test_the_hold_is_announced_once_for_a_fit_and_not_again_for_the_test_pass(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`BaseFinetuning` freezes from `setup`, which Lightning calls once per stage.

    Announcing it there printed "frozen until ..." a second time as the test pass
    began — about a run whose training had already finished, which reads as a run
    still holding its backbone. The freezing itself stays where Lightning puts it;
    only the sentence moved to `on_fit_start`.

    The moment is given in both currencies, as every boundary in `callbacks/` is.
    """
    data = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(4, 4), torch.tensor([0, 1, 0, 1])), batch_size=2
    )
    trained = _Trainable()
    trainer = quiet_trainer(max_epochs=2, callbacks=[Freeze(modules=["model"], until=1.0)])

    with caplog.at_level(logging.INFO):
        trainer.fit(trained, data)
        trainer.test(trained, data, verbose=False)

    said = [record.getMessage() for record in caplog.records if "Frozen until" in record.getMessage()]
    assert said == ["Frozen until epoch 2 (step 4): model"]
