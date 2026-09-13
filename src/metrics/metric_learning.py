"""Recall@k over an epoch of embeddings: how often a sample's nearest neighbours share its identity."""

from __future__ import annotations

import torch
import torchmetrics
from torch import Tensor
from torch.nn.functional import normalize
from torchmetrics.utilities.data import dim_zero_cat

from src.core import FEATURE_AXIS
from src.metrics.registry import metric_registry

QUERY_BLOCK = 1024
"""How many samples are ranked against the gallery at once.

Bounds the largest tensor alive in ``compute`` at this by the epoch rather than the epoch squared —
205 MB rather than 10 GB at fifty thousand samples — without changing what is read.
"""


@metric_registry.register("recall_at_k")
class RecallAtK(torchmetrics.Metric):
    """The share of samples whose ``k`` nearest neighbours in the epoch include one of their own identity.

    Deep metric learning calls this Recall@K. Information retrieval gives that name to a different
    number — the share of an identity's samples that were found, rather than whether any was — and
    calls this one HitRate@K. The registry writes the field's word; ``RetrievalHitRate`` is the
    library's equal of it and is what the tests score this against.

    Read it on validation and test. A gallery accumulated while the encoder is still moving holds
    vectors from several states of it, so the reading measures drift as much as separation; evaluation
    stages hold the model still. Confining a metric to a stage is not yet something a declaration says.

    Attributes:
        k: How far down the ranking a match still counts.
    """

    higher_is_better = True
    full_state_update = False

    # Declared for the type checker only; `add_state` is what creates these, resets them between
    # epochs and gathers them across devices.
    embeddings: list[Tensor]
    identities: list[Tensor]

    def __init__(self, k: int = 1) -> None:
        if k < 1:
            raise ValueError(f"A neighbour has to be looked for somewhere; 'k' was {k}.")
        super().__init__()
        self.k = k
        self.add_state("embeddings", default=[], dist_reduce_fx="cat")
        self.add_state("identities", default=[], dist_reduce_fx="cat")

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Keep this batch; what it is worth is not known until the rest of the epoch has arrived."""
        self.embeddings.append(predictions)
        self.identities.append(targets)

    def compute(self) -> Tensor:
        """Rank every sample against every other and count those that found one of their own.

        Normalized first, so the ranking follows angle rather than length whatever the model publishes.
        """
        embeddings = normalize(dim_zero_cat(self.embeddings), dim=FEATURE_AXIS)
        identities = dim_zero_cat(self.identities)
        self._refuse_a_gallery_smaller_than_the_reading(len(identities))
        found = [
            self._found_their_own(embeddings[at : at + QUERY_BLOCK], at, embeddings, identities)
            for at in range(0, len(identities), QUERY_BLOCK)
        ]
        return torch.cat(found).float().mean()

    def _found_their_own(self, queries: Tensor, at: int, gallery: Tensor, identities: Tensor) -> Tensor:
        """Whether each query in this block has one of its own among its ``k`` nearest in the gallery."""
        similarity = queries @ gallery.T
        # A query is its own nearest neighbour, always and perfectly; left in the ranking, an untrained
        # model reads 1.0. Its own column is its row plus where the block starts.
        rows = torch.arange(len(queries))
        similarity[rows, rows + at] = float("-inf")
        nearest = identities[similarity.topk(self.k, dim=-1).indices]
        return (nearest == identities[at : at + len(queries)].unsqueeze(-1)).any(dim=-1)

    def _refuse_a_gallery_smaller_than_the_reading(self, held: int) -> None:
        """Refused rather than answered about fewer neighbours than it was asked for.

        Looking only as far as the epoch reaches would answer a question nobody asked under the label of
        the one they did — ``recall_at_10`` reported as recall at however many there turned out to be.
        """
        if held <= self.k:
            raise ValueError(
                f"Recall at {self.k} ranks each sample against every other one of the epoch, and this "
                f"epoch held {held}. Lower 'k', or judge the run on a split with more in it."
            )
