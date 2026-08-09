"""One grammar for a task's loss: a single criterion is a list of one."""

from __future__ import annotations

import pytest
import torch

from src.assembly.tasks import _task_criterion
from src.config.tasks import LossConfig
from src.core import TargetFacts
from src.losses import CrossEntropyCriterion, ProxyAngularCriterion, WeightedSumCriterion

LOGITS = torch.tensor([[2.0, 0.5, 0.1]])
TARGET = torch.tensor([0])
EMBEDDING_DIM = 8  # offered to every loss; only one that names it receives it


def test_the_usual_declaration_costs_no_wrapper() -> None:
    """A plain ``loss: cross_entropy`` builds the criterion itself, not a sum of one."""
    built = _task_criterion(LossConfig(name="cross_entropy"), TargetFacts(), embedding_dim=EMBEDDING_DIM)

    assert isinstance(built, CrossEntropyCriterion)


def test_a_weight_on_a_single_loss_scales_it() -> None:
    """The list form could always carry a weight; the single form now says the same thing."""
    bare = _task_criterion(LossConfig(name="cross_entropy"), TargetFacts(), embedding_dim=EMBEDDING_DIM)
    halved = _task_criterion(LossConfig(name="cross_entropy", weight=0.5), TargetFacts(), embedding_dim=EMBEDDING_DIM)

    assert isinstance(halved, WeightedSumCriterion)
    assert halved(LOGITS, TARGET).total.item() == pytest.approx(0.5 * bare(LOGITS, TARGET).total.item())


def test_a_list_goes_through_the_same_weighted_sum() -> None:
    """List or single, the weight means the same thing — that is the point of one grammar."""
    bare = _task_criterion(LossConfig(name="cross_entropy"), TargetFacts(), embedding_dim=EMBEDDING_DIM)
    doubled = _task_criterion(
        [LossConfig(name="cross_entropy", weight=2.0)], TargetFacts(), embedding_dim=EMBEDDING_DIM
    )

    assert doubled(LOGITS, TARGET).total.item() == pytest.approx(2.0 * bare(LOGITS, TARGET).total.item())


def test_the_weight_never_reaches_the_constructor() -> None:
    """Declared as a field, so it cannot leak into the criterion's arguments."""
    assert LossConfig(name="cross_entropy", weight=0.5).params == {}


def test_derived_sizes_reach_a_proxy_criterion() -> None:
    """num_classes and embedding_dim come from assembly, never from config."""
    built = _task_criterion(LossConfig(name="arcface_proxy"), TargetFacts(num_classes=7), embedding_dim=EMBEDDING_DIM)

    assert isinstance(built, ProxyAngularCriterion)
    assert built.prototypes.shape == (7, EMBEDDING_DIM)
