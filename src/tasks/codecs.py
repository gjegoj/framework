"""Task-layer codecs: light shape/type adaptation of targets for loss/metrics.

Heavy decoding (label -> index, mask I/O) happens in the data layer; here we
only adjust the already-tensor target to the shape each objective's criterion
and metrics expect.
"""

from __future__ import annotations

from torch import Tensor

from src.core.entities import TargetView
from src.core.ports import TaskCodec


class MulticlassTaskCodec(TaskCodec):
    """Class-index targets for cross-entropy and accuracy: ``[B]`` long tensor."""

    def adapt(self, target: Tensor) -> TargetView:
        if target.ndim == 2 and target.size(-1) == 1:
            target = target.squeeze(-1)
        target = target.long()
        return TargetView(loss=target, metric=target)
