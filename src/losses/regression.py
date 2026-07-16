"""Regression criteria: MSE and L1 on raw outputs vs float targets."""

from __future__ import annotations

from typing import Any

from torch import nn

from src.losses.base import SingleTermCriterion
from src.losses.registry import criteria


@criteria.register("mse")
class MSECriterion(SingleTermCriterion):
    """Mean squared error on raw outputs vs float targets.

    Parameters:
        **kwargs: Forwarded verbatim to ``nn.MSELoss`` (``reduction``, ...).
    """

    component_name = "mse"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(nn.MSELoss(**kwargs))


@criteria.register("l1")
class L1Criterion(SingleTermCriterion):
    """Mean absolute error on raw outputs vs float targets.

    Parameters:
        **kwargs: Forwarded verbatim to ``nn.L1Loss`` (``reduction``, ...).
    """

    component_name = "l1"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(nn.L1Loss(**kwargs))
