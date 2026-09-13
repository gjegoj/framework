"""An objective that carries parameters of its own, standing in for any loss that does.

A shipped one exists — ``arcface_proxy`` keeps a prototype per identity — but what these tests pin is
the framework's rule that such a loss is optimized, checkpointed and restored with the run and stays
out of the network. Written here so that rule is not scored against one registered name, and because a
declaration reaches a class by import path.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.functional import cross_entropy

from src.core import LossOutput
from src.losses import Loss


class LearnedMargin(Loss):
    """Cross-entropy over logits this loss shifts by one margin per class, learned with the run.

    ``num_classes`` is a fact of the target rather than a declaration, so it arrives by signature the
    way every other derived fact does, and a run restating it is refused before the first step.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.margin = nn.Parameter(torch.zeros(num_classes))

    def forward(self, outputs: Tensor, targets: Tensor) -> LossOutput:
        return self.reported(cross_entropy(outputs + self.margin, targets))
