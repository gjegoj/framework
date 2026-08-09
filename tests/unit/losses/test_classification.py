"""Classification criteria: cross-entropy and binary cross-entropy."""

from __future__ import annotations

import pytest
import torch

from src.losses import BinaryCrossEntropyCriterion, CrossEntropyCriterion


def test_cross_entropy_returns_a_named_loss_with_grad() -> None:
    logits = torch.randn(4, 3, requires_grad=True)
    target = torch.tensor([0, 1, 2, 0])

    loss = CrossEntropyCriterion()(logits, target)

    assert set(loss.parts) == {"ce"}
    assert loss.total.requires_grad


def test_binary_cross_entropy_accepts_single_logit_heads() -> None:
    logits = torch.randn(4, 1, requires_grad=True)
    target = torch.tensor([0.0, 1.0, 1.0, 0.0])

    loss = BinaryCrossEntropyCriterion()(logits, target)

    assert set(loss.parts) == {"bce"}
    assert loss.total.shape == ()


def test_binary_cross_entropy_squeezes_the_channel_on_dense_shapes() -> None:
    logits = torch.randn(2, 1, 8, 8)
    target = torch.randint(0, 2, (2, 8, 8)).float()

    loss = BinaryCrossEntropyCriterion()(logits, target)

    assert loss.total.shape == ()


def test_binary_cross_entropy_handles_multilabel_shapes() -> None:
    logits = torch.randn(4, 5)
    target = torch.randint(0, 2, (4, 5)).float()

    loss = BinaryCrossEntropyCriterion()(logits, target)

    assert loss.total.shape == ()


def test_cross_entropy_matches_torch_reference() -> None:
    logits = torch.randn(8, 3)
    target = torch.randint(0, 3, (8,))

    loss = CrossEntropyCriterion()(logits, target)

    expected = torch.nn.functional.cross_entropy(logits, target)
    assert loss.total.item() == pytest.approx(expected.item())


def test_cross_entropy_forwards_any_torch_knob_via_kwargs() -> None:
    logits = torch.randn(8, 3)
    target = torch.randint(0, 3, (8,))

    loss = CrossEntropyCriterion(label_smoothing=0.2)(logits, target)

    expected = torch.nn.functional.cross_entropy(logits, target, label_smoothing=0.2)
    assert loss.total.item() == pytest.approx(expected.item())


def test_cross_entropy_converts_the_weight_list_to_a_tensor() -> None:
    logits = torch.randn(8, 3)
    target = torch.randint(0, 3, (8,))

    loss = CrossEntropyCriterion(weight=[1.0, 2.0, 0.5])(logits, target)

    expected = torch.nn.functional.cross_entropy(logits, target, weight=torch.tensor([1.0, 2.0, 0.5]))
    assert loss.total.item() == pytest.approx(expected.item())


def test_class_weights_live_in_a_buffer_and_move_with_the_module() -> None:
    criterion = CrossEntropyCriterion(weight=[1.0, 2.0, 0.5])

    assert any("weight" in name for name in dict(criterion.named_buffers()))


def test_binary_cross_entropy_converts_pos_weight_and_forwards_it() -> None:
    logits = torch.randn(6, 1)
    target = torch.randint(0, 2, (6,)).float()

    loss = BinaryCrossEntropyCriterion(pos_weight=[3.0])(logits, target)

    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.squeeze(-1), target, pos_weight=torch.tensor([3.0])
    )
    assert loss.total.item() == pytest.approx(expected.item())
