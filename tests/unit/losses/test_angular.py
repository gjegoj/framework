"""ArcFace: the margin math, and the proxy that trains an embedder with it."""

from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional

from src.core import Objective, TargetFacts
from src.losses import ArcFaceCriterion, ProxyAngularCriterion
from src.losses.angular import ArcFaceLoss
from src.losses.registry import criterion_registry
from src.tasks.registry import objective_registry

COSINES = torch.tensor([[0.8, 0.1, -0.3], [0.2, 0.6, 0.1]])
LABELS = torch.tensor([0, 1])


def test_zero_margin_is_cross_entropy_on_scaled_cosines() -> None:
    """The identity that pins the formula: no margin, no difference."""
    value = ArcFaceLoss(margin=0.0, scale=10.0)(COSINES, LABELS)

    assert value.item() == pytest.approx(functional.cross_entropy(COSINES * 10.0, LABELS).item(), abs=1e-6)


def test_the_margin_makes_the_right_answer_cost_more() -> None:
    """The target must win by the margin, not merely win — so the same cosines lose value."""
    plain = ArcFaceLoss(margin=0.0)(COSINES, LABELS)
    pushed = ArcFaceLoss(margin=0.5)(COSINES, LABELS)

    assert pushed.item() > plain.item()


def test_an_annealed_margin_acts_on_the_next_step() -> None:
    """Cos/sin are derived per forward: a cached construction would make `anneal` a no-op."""
    criterion = ArcFaceLoss(margin=0.0)
    before = criterion(COSINES, LABELS)

    criterion.margin = 0.5
    after = criterion(COSINES, LABELS)

    assert after.item() != pytest.approx(before.item())


def test_the_worst_angle_keeps_a_finite_gradient() -> None:
    """Past pi - m the substitution loses monotonicity; the linear fallback covers it."""
    hopeless = torch.tensor([[-0.999, 0.9]], requires_grad=True)

    ArcFaceLoss(margin=0.5)(hopeless, torch.tensor([0])).backward()

    assert hopeless.grad is not None and torch.isfinite(hopeless.grad).all()


def test_raw_scores_are_refused_with_directions() -> None:
    """Silently clamping linear logits into [-1, 1] would train something wrong quietly."""
    with pytest.raises(ValueError, match="arcface_proxy"):
        ArcFaceLoss()(torch.tensor([[3.2, -7.0]]), torch.tensor([0]))


@pytest.mark.parametrize(("kwargs", "named"), [({"margin": math.pi}, "margin"), ({"scale": 0.0}, "scale")])
def test_an_argument_outside_its_domain_is_refused(kwargs: dict[str, float], named: str) -> None:
    with pytest.raises(ValueError, match=named):
        ArcFaceLoss(**kwargs)


def test_the_proxy_trains_an_embedder_end_to_end() -> None:
    """Gradients must reach both the embedding and the prototypes it is judged against."""
    embeddings = torch.randn(4, 8, requires_grad=True)
    proxy = ProxyAngularCriterion(num_classes=3, embedding_dim=8)

    proxy(embeddings, torch.tensor([0, 1, 2, 0])).total.backward()

    assert embeddings.grad is not None and torch.isfinite(embeddings.grad).all()
    assert proxy.prototypes.grad is not None and bool(proxy.prototypes.grad.abs().sum() > 0)


def test_the_prototypes_are_trained_weights_so_they_checkpoint() -> None:
    """Unlike focal's alpha or expectation's class values, these are learned — a resume needs them."""
    assert "prototypes" in ProxyAngularCriterion(num_classes=3, embedding_dim=8).state_dict()


def test_the_proxy_logs_under_its_margins_name() -> None:
    """Pure plumbing: swapping the margin renames the logged part honestly."""
    loss = ProxyAngularCriterion(num_classes=3, embedding_dim=8)(torch.randn(2, 8), torch.tensor([0, 1]))

    assert set(loss.parts) == {"arcface"}


def test_an_embedding_of_the_wrong_width_is_reported_with_both_numbers() -> None:
    with pytest.raises(ValueError, match="8"):
        ProxyAngularCriterion(num_classes=3, embedding_dim=8)(torch.randn(2, 16), torch.tensor([0, 1]))


def test_margin_arguments_reach_the_default_inner() -> None:
    """The expectation slot pattern: kwargs configure the default, or an inner replaces it."""
    proxy = ProxyAngularCriterion(num_classes=3, embedding_dim=8, margin=0.0, scale=1.0)
    embeddings = torch.randn(2, 8)

    cosines = functional.linear(functional.normalize(embeddings, dim=1), functional.normalize(proxy.prototypes, dim=1))

    expected = functional.cross_entropy(cosines, LABELS)
    assert proxy(embeddings, LABELS).total.item() == pytest.approx(expected.item(), abs=1e-6)


def test_an_inner_beside_its_arguments_is_refused() -> None:
    with pytest.raises(ValueError, match="not both"):
        ProxyAngularCriterion(num_classes=3, embedding_dim=8, inner=ArcFaceCriterion(), margin=0.3)


def test_a_metric_task_that_declared_labels_receives_them() -> None:
    """Facts decide: encoded labels reach the criterion, structure-supervised tasks stay targetless."""
    behaviour = objective_registry.create(Objective.METRIC)

    assert behaviour.build_target_adapter(TargetFacts(num_classes=5)) is not None
    assert behaviour.build_target_adapter(TargetFacts()) is None


def test_both_are_reachable_from_config_by_name() -> None:
    assert isinstance(criterion_registry.create("arcface", margin=0.3), ArcFaceCriterion)
    assert isinstance(criterion_registry.create("arcface_proxy", num_classes=3, embedding_dim=8), ProxyAngularCriterion)
