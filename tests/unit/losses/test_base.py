"""``WrappedCriterion``: wrapping any torch loss module is a three-line subclass."""

from __future__ import annotations

from typing import ClassVar

import pytest
import torch
from torch import Tensor, nn

from src.losses import WrappedCriterion


class ScaledLoss(nn.Module):
    """A custom loss with a learnable parameter, standing in for any exotic loss."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return self.scale * (logits - target).abs().mean()


class ScaledCriterion(WrappedCriterion):
    part_name: ClassVar[str] = "scaled"

    def __init__(self) -> None:
        super().__init__(ScaledLoss())


def test_subclass_wraps_a_torch_loss_under_its_part_name() -> None:
    loss = ScaledCriterion()(torch.zeros(4), torch.ones(4))

    assert set(loss.parts) == {"scaled"}
    assert loss.total.item() == 2.0


def test_wrapped_module_is_registered_for_training_and_devices() -> None:
    criterion = ScaledCriterion()

    assert any(isinstance(module, ScaledLoss) for module in criterion.modules())
    assert sum(1 for _ in criterion.parameters()) == 1


def test_a_subclass_that_forgot_its_part_name_is_refused_where_it_is_built() -> None:
    """``part_name`` is annotated without a value, so forgetting it is a live possibility.

    Unchecked, the omission surfaced as an ``AttributeError`` inside the first
    ``forward`` — a thousand steps into a run, reading as a torch problem rather than as
    a missing declaration. A criterion is built at assembly, so that is where this fires:
    before a single batch is read.
    """

    class Nameless(WrappedCriterion):
        def __init__(self) -> None:
            super().__init__(nn.L1Loss())

    with pytest.raises(TypeError, match="declares no 'part_name'"):
        Nameless()
