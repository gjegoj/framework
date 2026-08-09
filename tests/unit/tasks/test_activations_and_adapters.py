"""Public activations and target adapters: reusable bricks, open to any callable."""

from __future__ import annotations

import torch

from src.core import AdaptedTarget
from src.losses import MeanSquaredErrorCriterion
from src.models import LinearHead, TaskComponents
from src.tasks.activations import (
    sigmoid_probabilities,
    softmax_probabilities,
    squeeze_single_output,
)
from src.tasks.adapters import as_class_indices, as_is, float_for_loss


def test_softmax_probabilities_sum_to_one() -> None:
    probabilities = softmax_probabilities(torch.randn(4, 3))

    assert torch.allclose(probabilities.sum(dim=1), torch.ones(4))


def test_sigmoid_probabilities_squeeze_single_logit_heads() -> None:
    assert sigmoid_probabilities(torch.zeros(4, 1)).shape == (4,)
    assert sigmoid_probabilities(torch.zeros(4, 5)).shape == (4, 5)


def test_squeeze_single_output_leaves_wide_outputs_alone() -> None:
    assert squeeze_single_output(torch.zeros(4, 1)).shape == (4,)
    assert squeeze_single_output(torch.zeros(4, 3)).shape == (4, 3)


def test_as_is_keeps_both_views_untouched() -> None:
    target = torch.tensor([1, 2])

    adapted = as_is(target)

    assert adapted.for_loss is target
    assert adapted.for_metrics is target


def test_as_class_indices_makes_both_views_long() -> None:
    """Augmented masks come back as int32; class indices must be long for both consumers."""
    adapted = as_class_indices(torch.tensor([[0, 2]], dtype=torch.int32))

    assert adapted.for_loss.dtype == torch.long
    assert adapted.for_metrics.dtype == torch.long


def test_float_for_loss_floats_only_the_loss_view() -> None:
    adapted = float_for_loss(torch.tensor([0, 1]))

    assert adapted.for_loss.dtype == torch.float32
    assert adapted.for_metrics.dtype == torch.long


def test_any_callable_works_as_an_activation() -> None:
    """The Activation port is a plain callable: torch.tanh plugs in as-is."""
    components = TaskComponents(
        head=LinearHead(8, 1),
        criterion=MeanSquaredErrorCriterion(),
        activation=torch.tanh,
        target_adapter=as_is,
    )

    predictions = components.activation(torch.zeros(2, 1))

    assert torch.equal(predictions, torch.zeros(2, 1))


def test_custom_adapter_is_a_plain_function() -> None:
    """The TargetAdapter port is a plain callable too."""

    def clamp_for_loss(target: torch.Tensor) -> AdaptedTarget:
        return AdaptedTarget(for_loss=target.clamp(0, 1), for_metrics=target)

    adapted = clamp_for_loss(torch.tensor([-1.0, 2.0]))

    assert adapted.for_loss.tolist() == [0.0, 1.0]
