"""Learning from the batch itself: two draws of one sample belong together, and every other pair does not."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn.functional import cross_entropy, normalize

from src.core import LossOutput
from src.losses.base import Loss
from src.losses.registry import loss_registry

PAIR = 2
"""How many draws this objective reads. A third has no place in a similarity matrix of two sides."""


@loss_registry.register("info_nce")
class InfoNce(Loss):
    """Two views of a sample against every other sample's, both ways round — the CLIP objective.

    What supervises it is the batch: the draw that came from the same picture is the right answer and
    the rest of the batch is the wrong one, so the target is which row a sample is, and a run needs no
    column for it. Both directions are scored and averaged, because a matrix read one way only makes
    one side of every pair responsible for the match.

    The comparison is between *directions*: the vectors are normalised here rather than by whatever
    produced them. A backbone that normalised would be answering for an objective it does not know it
    has, and the same features feed a head, a metric and an export that each want them as they are.

    The temperature is learned, as CLIP's is, and kept as the log of the scale so that it stays
    positive without being clamped. It is a parameter of the objective, so it is optimized and written
    into the checkpoint with the run rather than beside it — the arrangement an angular margin already
    uses here.

    Parameters:
        temperature: What the similarities are divided by before they are read as a distribution; the
            value the scale starts from, and a run learns on from there.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        if temperature <= 0:
            raise ValueError(f"A temperature divides the similarities, so it is positive; got {temperature}.")
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))

    def forward(self, outputs: Tensor, targets: Tensor) -> LossOutput:
        first, second = self._two_draws(outputs, len(targets))
        similarity = normalize(first, dim=-1) @ normalize(second, dim=-1).T * self.log_scale.exp()
        both_ways = cross_entropy(similarity, targets) + cross_entropy(similarity.T, targets)
        return self.reported(both_ways / PAIR)

    def _two_draws(self, outputs: Tensor, samples: int) -> tuple[Tensor, Tensor]:
        """The pair to compare, recovered from the rows alone, or a refusal naming what the run draws.

        Nothing declares the number of draws here. The target says how many samples the batch holds and
        the rows say how many answers came back, and a viewing backbone leaves every draw of a sample
        next to the next — so the two are the halves of one unflattening, and a run drawing some other
        number is a run this objective cannot read.
        """
        if outputs.ndim != 2 or outputs.shape[0] != samples * PAIR:
            raise ValueError(
                f"This objective reads two draws of every sample: {samples} samples should have answered "
                f"with {samples * PAIR} rows, and it was handed {list(outputs.shape)}. A stage draws them "
                f"with `MultiViewTransform(views: {PAIR})` and a `multiview` backbone folds them in."
            )
        drawn = outputs.unflatten(0, (samples, PAIR))
        return drawn[:, 0], drawn[:, 1]
