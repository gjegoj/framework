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
    """A sample's two answers against every other sample's, both ways round — the CLIP objective.

    What supervises it is the batch: the answer that came from the same sample is the right one and the
    rest of the batch is wrong, so the target is which row a sample is, and a run needs no column for
    it. Both directions are scored and averaged, because a matrix read one way only makes one side of
    every pair responsible for the match.

    Where the two answers came from is not this objective's business and is deliberately not asked.
    Drawing one picture twice and pairing a picture with its caption arrive here as the same thing —
    twice as many rows as the batch has samples, a sample's own adjacent — which is why the pairing
    this library is named for needed nothing added here.

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

    reads_per_sample: int = PAIR
    """Two, which is what `_two_draws` recovers below and what a head declared over a pair is checked against."""

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
        """The pair to compare, recovered from the rows alone, or a refusal naming both ways to make one.

        Nothing declares the number of answers here. The target says how many samples the batch holds
        and the rows say how many answers came back, and whatever produced them left a sample's own next
        to each other — so the two are the halves of one unflattening, and a run answering some other
        number of times is a run this objective cannot read.
        """
        if outputs.ndim != 2 or outputs.shape[0] != samples * PAIR:
            raise ValueError(
                f"This objective reads two answers for every sample: {samples} samples should have "
                f"answered with {samples * PAIR} rows, and it was handed {list(outputs.shape)}. Two come "
                f"either from one input drawn twice — `MultiViewTransform(views: {PAIR})` under a "
                f"`multiview` backbone — or from two inputs read side by side, a `multiencoder` backbone "
                f"under a head declared over both its streams."
            )
        drawn = outputs.unflatten(0, (samples, PAIR))
        return drawn[:, 0], drawn[:, 1]
