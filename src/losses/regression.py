"""Objectives over numbers: a distance to the wanted value, or to the value a distribution stands for."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import torch
from torch import Tensor, nn

from src.core import CLASS_AXIS, LossOutput
from src.losses.base import Loss, TorchLoss
from src.losses.registry import loss_registry


@loss_registry.register("mse")
class MeanSquaredError(TorchLoss):
    module_type: ClassVar[type[nn.Module]] = nn.MSELoss
    squeezes_channel: ClassVar[bool] = True


@loss_registry.register("mae")
class MeanAbsoluteError(TorchLoss):
    module_type: ClassVar[type[nn.Module]] = nn.L1Loss
    squeezes_channel: ClassVar[bool] = True


@loss_registry.register("huber")
class Huber(TorchLoss):
    """Squared close to the wanted value, absolute far from it: an outlier stops steering the fit."""

    module_type: ClassVar[type[nn.Module]] = nn.HuberLoss
    squeezes_channel: ClassVar[bool] = True


@loss_registry.register("smooth_l1")
class SmoothL1(TorchLoss):
    module_type: ClassVar[type[nn.Module]] = nn.SmoothL1Loss
    squeezes_channel: ClassVar[bool] = True


@loss_registry.register("expectation")
class Expectation(Loss):
    """The distance between the numbers two distributions over bins stand for.

    A binned regression learns a distribution, but is judged on a number, and cross-entropy alone does
    not care which number its distribution averages to. This term does: it weighs both sides by the
    value each bin stands for and compares what comes out. `values` is a fact of the target encoder
    that laid the bins out, so a run never writes it.

    Args:
        values: The number each bin stands for, in bin order.
    """

    values: Tensor
    """Registered as a buffer, so the bins travel with the model to whatever device it runs on."""

    def __init__(self, values: Sequence[float], distance: str = "mae") -> None:
        super().__init__()
        if len(values) < 2:
            raise ValueError(f"An expectation spans at least two bins; {len(values)} were declared.")
        self.register_buffer("values", torch.as_tensor(list(values), dtype=torch.float))
        self.distance: Loss = loss_registry.get(distance)()

    def forward(self, outputs: Tensor, targets: Tensor) -> LossOutput:
        bins = self.values.numel()
        if outputs.shape[CLASS_AXIS] != bins:
            raise ValueError(
                f"An expectation over {bins} bins reads a score per bin, but the output has "
                f"{outputs.shape[CLASS_AXIS]}. The head is sized from the encoder that laid them out."
            )
        predicted = outputs.softmax(dim=CLASS_AXIS) @ self.values
        wanted = targets.float() @ self.values
        return self.reported(self.distance(predicted, wanted).total)
