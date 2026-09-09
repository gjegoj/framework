"""One grammar for a task's loss: a single criterion is a list of one."""

from __future__ import annotations

import pytest
import torch

from src.config.tasks import LossConfig
from src.core import TaskFacts
from src.losses import CrossEntropyCriterion, ExpectationCriterion, ProxyAngularCriterion, WeightedSumCriterion
from src.losses.build import build_criterion

LOGITS = torch.tensor([[2.0, 0.5, 0.1]])
TARGET = torch.tensor([0])
EMBEDDING_DIM = 8  # a fact of the stream the task reads; a criterion sized by it says so


def test_the_usual_declaration_costs_no_wrapper() -> None:
    """A plain ``loss: cross_entropy`` builds the criterion itself, not a sum of one."""
    built = build_criterion(LossConfig(name="cross_entropy"), TaskFacts(), embedding_dim=EMBEDDING_DIM)

    assert isinstance(built, CrossEntropyCriterion)


def test_a_weight_on_a_single_loss_scales_it() -> None:
    """The list form could always carry a weight; the single form now says the same thing."""
    bare = build_criterion(LossConfig(name="cross_entropy"), TaskFacts(), embedding_dim=EMBEDDING_DIM)
    halved = build_criterion(LossConfig(name="cross_entropy", weight=0.5), TaskFacts(), embedding_dim=EMBEDDING_DIM)

    assert isinstance(halved, WeightedSumCriterion)
    assert halved(LOGITS, TARGET).total.item() == pytest.approx(0.5 * bare(LOGITS, TARGET).total.item())


def test_a_list_goes_through_the_same_weighted_sum() -> None:
    """List or single, the weight means the same thing — that is the point of one grammar."""
    bare = build_criterion(LossConfig(name="cross_entropy"), TaskFacts(), embedding_dim=EMBEDDING_DIM)
    doubled = build_criterion([LossConfig(name="cross_entropy", weight=2.0)], TaskFacts(), embedding_dim=EMBEDDING_DIM)

    assert doubled(LOGITS, TARGET).total.item() == pytest.approx(2.0 * bare(LOGITS, TARGET).total.item())


def test_the_weight_never_reaches_the_constructor() -> None:
    """Declared as a field, so it cannot leak into the criterion's arguments."""
    assert LossConfig(name="cross_entropy", weight=0.5).params == {}


def test_derived_sizes_reach_a_proxy_criterion() -> None:
    """num_classes and embedding_dim come from the task's facts, never from config."""
    built = build_criterion(LossConfig(name="arcface_proxy"), TaskFacts(num_classes=7), embedding_dim=EMBEDDING_DIM)

    assert isinstance(built, ProxyAngularCriterion)
    assert built.prototypes.shape == (7, EMBEDDING_DIM)


def test_a_criterion_sized_by_the_facts_says_so_on_its_class() -> None:
    """The explicit protocol: ``sized`` is how a class declares it takes facts, greppable, no name-matching."""
    built = ProxyAngularCriterion.sized(TaskFacts(num_classes=7), EMBEDDING_DIM)

    assert built.prototypes.shape == (7, EMBEDDING_DIM)


def test_bin_values_reach_the_expectation_term() -> None:
    built = build_criterion(
        LossConfig(name="expectation"), TaskFacts(class_values=(0.0, 1.0, 2.0)), embedding_dim=EMBEDDING_DIM
    )

    assert isinstance(built, ExpectationCriterion)
    assert built.class_values.tolist() == [0.0, 1.0, 2.0]


def test_a_fact_written_in_config_is_refused_by_name() -> None:
    """Nothing wins silently: a size the data decides is not a knob, and the declaration says which key to drop."""
    with pytest.raises(ValueError, match="num_classes"):
        build_criterion(
            LossConfig(name="arcface_proxy", num_classes=3), TaskFacts(num_classes=7), embedding_dim=EMBEDDING_DIM
        )


def test_a_sized_criterion_without_its_fact_is_refused_naming_it() -> None:
    with pytest.raises(LookupError, match="num_classes"):
        build_criterion(LossConfig(name="arcface_proxy"), TaskFacts(), embedding_dim=EMBEDDING_DIM)


def test_a_raw_torch_loss_by_target_is_refused_at_build_naming_the_wrap() -> None:
    """``nn.MSELoss`` returns a bare tensor where a criterion returns a ``Loss`` with a named part;
    unrefused, it built and died in the first training step."""
    with pytest.raises(TypeError, match=r"MSELoss.*Criterion|Criterion.*MSELoss"):
        build_criterion(LossConfig.model_validate({"_target_": "torch.nn.MSELoss"}))
