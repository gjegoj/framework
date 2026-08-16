"""Focal loss: cross-entropy whose well-classified examples fade away."""

from __future__ import annotations

import pytest
import torch
from torch.nn.functional import one_hot

from src.core import Batch, DataProfile, Objective, OutputTopology, TargetFacts, Task
from src.losses import CrossEntropyCriterion, FocalCriterion
from src.losses.classification import FocalLoss
from src.losses.registry import criterion_registry
from src.transforms.batch import MixUp
from tests.support.narrowing import tensor

LOGITS = torch.tensor([[4.0, 0.0, 0.0], [0.5, 0.4, 0.3]])  # one easy sample, one hard
TARGET = torch.tensor([0, 0])


def test_gamma_zero_is_exactly_cross_entropy() -> None:
    """The one identity that pins the whole formula to the paper's."""
    focal = FocalCriterion(gamma=0.0)(LOGITS, TARGET)
    ce = CrossEntropyCriterion()(LOGITS, TARGET)

    assert focal.total.item() == pytest.approx(ce.total.item(), abs=1e-6)


def test_easy_examples_fade_harder_than_hard_ones() -> None:
    """The point of gamma: the well-classified sample loses its say first."""
    plain = FocalLoss(gamma=0.0, reduction="none")(LOGITS, TARGET)
    focused = FocalLoss(gamma=2.0, reduction="none")(LOGITS, TARGET)
    kept = focused / plain  # per-sample survival factor

    assert kept[0].item() < kept[1].item()


def test_alpha_reweights_the_classes() -> None:
    doubled = FocalLoss(alpha=[2.0, 1.0, 1.0], gamma=0.0, reduction="none")(LOGITS, TARGET)
    plain = FocalLoss(gamma=0.0, reduction="none")(LOGITS, TARGET)

    assert torch.allclose(doubled, 2.0 * plain)


def test_one_module_serves_dense_shapes() -> None:
    """The class dim is dim 1 and the pixels ride along — segmentation needs no second focal."""
    logits = torch.randn(2, 3, 4, 4)
    mask = torch.randint(0, 3, (2, 4, 4))

    focal = FocalCriterion(gamma=0.0)(logits, mask)
    ce = CrossEntropyCriterion()(logits, mask)

    assert focal.total.item() == pytest.approx(ce.total.item(), abs=1e-6)


def test_a_one_hot_soft_target_matches_the_hard_one_exactly() -> None:
    """The soft branch is a generalisation, not a second loss: one-hot reduces to gather."""
    soft = one_hot(TARGET, 3).float()

    assert FocalLoss(alpha=[2.0, 1.0, 1.0], reduction="none")(LOGITS, soft).allclose(
        FocalLoss(alpha=[2.0, 1.0, 1.0], reduction="none")(LOGITS, TARGET)
    )


def test_a_mixed_batch_reaches_the_loss_and_trains() -> None:
    """MixUp hands multiclass losses a distribution; focal must take it as CE does."""
    torch.manual_seed(0)
    profile = DataProfile()
    profile.record("label", TargetFacts(num_classes=3))
    task = Task(name="label", output_topology=OutputTopology.GLOBAL, objective=Objective.MULTICLASS, metrics={})
    batch = Batch(inputs={"image": torch.randn(4, 3, 4, 4)}, targets={"label": torch.tensor([0, 1, 2, 0])})
    mixed = MixUp([task], profile)(batch)
    logits = torch.randn(4, 3, requires_grad=True)

    loss = FocalCriterion(gamma=2.0)(logits, tensor(mixed.targets["label"]))
    loss.total.backward()

    assert torch.isfinite(loss.total)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_a_fully_learned_sample_keeps_a_finite_gradient() -> None:
    """p_t saturates to exactly 1.0 in fp32; pow's backward at zero base is infinite for gamma < 1."""
    saturated = torch.tensor([[40.0, 0.0]], requires_grad=True)

    FocalLoss(gamma=0.5)(saturated, torch.tensor([0])).backward()

    assert saturated.grad is not None and torch.isfinite(saturated.grad).all()


def test_an_alpha_of_the_wrong_length_is_reported_with_both_sizes() -> None:
    """A short alpha would work for low classes and crash mid-epoch on the first high one."""
    with pytest.raises(ValueError, match="2 alpha weight"):
        FocalLoss(alpha=[1.0, 1.0])(LOGITS, TARGET)


def test_alpha_stays_out_of_the_checkpoint() -> None:
    """It describes the recipe, not the trained weights; a later run may weigh differently."""
    assert "alpha" not in FocalLoss(alpha=[1.0, 2.0, 3.0]).state_dict()


@pytest.mark.parametrize(
    ("kwargs", "named"),
    [({"gamma": -1.0}, "gamma"), ({"eps": 0.0}, "eps"), ({"reduction": "median"}, "reduction")],
)
def test_an_argument_outside_its_domain_is_refused(kwargs: dict[str, float | str], named: str) -> None:
    """The message has to name the argument, or a user reads it without learning what to change.

    Case-insensitive: a choice is refused by the ``Literal`` alias that declares
    it, and an alias is named after its parameter with a capital.
    """
    with pytest.raises(ValueError, match=f"(?i){named}"):
        FocalLoss(**kwargs)  # type: ignore[arg-type]


def test_it_is_reachable_from_config_by_name() -> None:
    assert isinstance(criterion_registry.create("focal", gamma=1.5), FocalCriterion)


def test_it_logs_under_its_own_name() -> None:
    assert set(FocalCriterion()(LOGITS, TARGET).parts) == {"focal"}
