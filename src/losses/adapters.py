"""Use ordinary PyTorch losses through the framework's reporting contract."""

from typing import cast

from torch import Tensor, nn

from src.core import LossOutput
from src.losses.base import Loss, LossInput


class TorchLoss(Loss):
    def __init__(self, criterion: nn.Module, *, log_name: str) -> None:
        super().__init__()
        self.criterion = criterion
        self.log_name = log_name

    def forward(self, inputs: LossInput) -> LossOutput:
        value = cast(Tensor, self.criterion(inputs.outputs, inputs.targets))
        return LossOutput(total=value, losses={self.log_name: value}, contributions={self.log_name: value})
