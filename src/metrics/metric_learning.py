"""What an epoch of directions says about itself: readings taken by comparing a gallery with itself."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar, override

import torch
import torchmetrics
from torch import Tensor
from torch.nn.functional import normalize
from torchmetrics.utilities.data import dim_zero_cat

from src.core import FEATURE_AXIS, Stage
from src.metrics.registry import metric_registry

QUERY_BLOCK = 1024
"""How many samples are ranked against the gallery at once.

Bounds the largest tensor alive in ``compute`` at this by the epoch rather than the epoch squared —
205 MB rather than 10 GB at fifty thousand samples — without changing what is read.
"""

LEVELS = 2001
"""How many thresholds a verification reading tries, spread evenly over the whole cosine range.

A threshold is looked for on a grid rather than among the values themselves because the values are
every pair of the epoch — quadratic, and past sorting long before the ranking above is past blocking.
What the grid costs is *which* threshold is found, never what is reported about it: measured against an
exact sweep over six galleries, a grid of this many gave up at most 2.23e-05 of accuracy and one of 201
up to 1.34e-04, while a step of 0.001 is finer than any deployment sets a cosine to.
"""


@dataclass(frozen=True, slots=True)
class Separation:
    """One threshold over an epoch's pairs and what it reads there — the two are only true together."""

    threshold: Tensor
    accuracy: Tensor


class GalleryReading(torchmetrics.Metric, ABC):
    """A reading taken by comparing one epoch of directions with each other, rather than with a target.

    The shape every retrieval reading shares: keep the epoch, normalize it, rank each sample against
    every other one, and never let a sample be its own neighbour. Written once because the block
    arithmetic below is the kind of thing that is wrong by a little and looks like nothing, and three
    copies of it would be three chances at that.

    Read on validation and test, as ``read_on`` declares, and declared here rather than on each reading:
    what a gallery is meaningful in follows from its being a gallery. A gallery accumulated while the
    encoder is still moving holds vectors from several states of it, so the reading would measure drift
    as much as separation; evaluation stages hold the model still.
    """

    read_on: ClassVar[frozenset[Stage]] = frozenset({Stage.VAL, Stage.TEST})
    higher_is_better: bool | None = True
    full_state_update = False

    # Declared for the type checker only; `add_state` is what creates these, resets them between
    # epochs and gathers them across devices. Measured on torchmetrics 1.9.0: readings whose state
    # agrees after a batch become one compute group, after which only the group's leader is updated and
    # the rest hold its state by reference — so a run declaring four of these keeps one epoch, not four.
    embeddings: list[Tensor]
    identities: list[Tensor]

    def __init__(self) -> None:
        super().__init__()
        self.add_state("embeddings", default=[], dist_reduce_fx="cat")
        self.add_state("identities", default=[], dist_reduce_fx="cat")

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Keep this batch; what it is worth is not known until the rest of the epoch has arrived."""
        self.embeddings.append(predictions)
        self.identities.append(targets)

    def compute(self) -> Tensor:
        """The epoch as one gallery, normalized so ranking follows angle whatever the model publishes."""
        embeddings = normalize(dim_zero_cat(self.embeddings), dim=FEATURE_AXIS)
        identities = dim_zero_cat(self.identities)
        self._refuse_a_gallery_of_one_identity(identities)
        return self._read(embeddings, identities)

    @abstractmethod
    def _read(self, embeddings: Tensor, identities: Tensor) -> Tensor:
        """What this reading asks of a gallery it has been handed whole."""

    def _blocks(self, embeddings: Tensor, identities: Tensor) -> Iterator[tuple[Tensor, Tensor, int]]:
        """Each block of the gallery against the whole of it: how alike, who they are, and where it starts.

        Where it starts is the third of those because a sample's own column sits at its row plus that
        offset, and what to do about it is the reading's to say — ranking puts it last, counting pairs
        drops it. Nothing here decides that; getting the offset wrong excludes somebody else's sample
        from every block but the first, which changes a reading by a little and looks like nothing.
        """
        for at in range(0, len(identities), QUERY_BLOCK):
            queries = embeddings[at : at + QUERY_BLOCK]
            yield queries @ embeddings.T, identities[at : at + len(queries)], at

    def _ranked(self, embeddings: Tensor, identities: Tensor) -> Iterator[tuple[Tensor, Tensor]]:
        """Each block of queries against the whole gallery, with nobody left to retrieve themselves.

        A query is its own nearest neighbour, always and perfectly; left in the ranking, an untrained
        model reads a perfect score. Negative infinity puts that column last in every ordering taken
        from these numbers, which is what a ranking needs and what a count of pairs cannot use.
        """
        for similarity, mine, at in self._blocks(embeddings, identities):
            rows = torch.arange(len(mine), device=similarity.device)
            similarity[rows, rows + at] = float("-inf")
            yield similarity, mine

    def _refuse_a_gallery_of_one_identity(self, identities: Tensor) -> None:
        """A gallery of one identity makes every neighbour a match, so the reading cannot fail.

        Measured on an untrained encoder over 148 samples: recall@1 reads 1.0000 over one identity,
        0.5878 over two, 0.1959 over five and 0.0541 over 37 — the reading follows how many identities
        the gallery holds at least as much as it follows the model. One identity is the case that is
        degenerate by construction, and the only place a floor can be drawn without choosing a number
        nobody can defend; that the baseline moves with the rest is said where a run reads them.
        """
        held = int(identities.unique().numel())
        if held < 2:
            raise ValueError(
                f"This epoch held {len(identities)} samples of one identity, and a reading taken by "
                "ranking them against each other is then 1.0 whatever the model learned: every "
                "neighbour is a match. Judge the run on a split holding more than one identity."
            )

    def _refuse_an_epoch_where_no_identity_repeats(self, identities: Tensor) -> None:
        """For a reading about matches found: with nobody to find, its best is reached by finding nobody.

        An average precision over no query at all is a mean of nothing, and a threshold over pairs that
        are every one of them strangers is perfect as soon as it rejects them all. Declared here, next
        to the refusal every gallery owes, and called by the readings it is true of — recall reads 0.0
        on such an epoch, which is its floor rather than its ceiling and is a true thing to report.
        """
        if int(identities.unique(return_counts=True)[1].max()) < 2:
            raise ValueError(
                "No picture in this epoch has another of its own identity, so there is nothing for a "
                "reading about matches to have got right or wrong. Judge the run on a split whose "
                "identities repeat."
            )


@metric_registry.register("recall_at_k")
class RecallAtK(GalleryReading):
    """The share of samples whose ``k`` nearest neighbours in the epoch include one of their own identity.

    Deep metric learning calls this Recall@K. Information retrieval gives that name to a different
    number — the share of an identity's samples that were found, rather than whether any was — and
    calls this one HitRate@K. The registry writes the field's word; ``RetrievalHitRate`` is the
    library's equal of it and is what the tests score this against.

    Attributes:
        k: How far down the ranking a match still counts.
    """

    def __init__(self, k: int = 1) -> None:
        if k < 1:
            raise ValueError(f"A neighbour has to be looked for somewhere; 'k' was {k}.")
        super().__init__()
        self.k = k

    @override
    def _read(self, embeddings: Tensor, identities: Tensor) -> Tensor:
        """Rank every sample against every other and count those that found one of their own."""
        self._refuse_a_gallery_smaller_than_the_reading(len(identities))
        found = [
            (identities[similarity.topk(self.k, dim=-1).indices] == mine.unsqueeze(-1)).any(dim=-1)
            for similarity, mine in self._ranked(embeddings, identities)
        ]
        return torch.cat(found).float().mean()

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


@metric_registry.register("map")
class MeanAveragePrecision(GalleryReading):
    """How well every picture of a query's identity is ranked, not merely whether one of them is first.

    Recall asks whether a match turned up in the first ``k``; this asks about all of them, which is the
    question wherever an identity has many pictures — a model that finds one of six and buries the rest
    reads the same as one that finds all six under recall, and differently here. ``RetrievalMAP`` is the
    library's equal of it and is what the tests score this against.

    A query whose identity appears nowhere else in the epoch is **left out** rather than scored zero:
    there was nothing for it to rank well, and scoring it would make the reading follow how many one-off
    identities the split happened to hold rather than the model. An epoch in which nobody has a match is
    refused, because a mean over no queries is not a reading of anything.
    """

    @override
    def _read(self, embeddings: Tensor, identities: Tensor) -> Tensor:
        self._refuse_an_epoch_where_no_identity_repeats(identities)
        scored = [
            self._averaged(similarity, mine, identities) for similarity, mine in self._ranked(embeddings, identities)
        ]
        return torch.cat(scored).mean()

    def _averaged(self, similarity: Tensor, mine: Tensor, identities: Tensor) -> Tensor:
        """The average precision of every query in this block that has somebody of its own to find.

        The self column sits at negative infinity, so it sorts last and is dropped whole rather than
        counted as a match at the bottom of every ranking.
        """
        order = similarity.argsort(dim=-1, descending=True)[:, :-1]
        hits = (identities[order] == mine.unsqueeze(-1)).float()
        found = hits.cumsum(dim=-1)
        depth = torch.arange(1, hits.size(-1) + 1, device=hits.device, dtype=hits.dtype)
        precision = found / depth
        theirs = hits.sum(dim=-1)
        averaged = (precision * hits).sum(dim=-1) / theirs.clamp(min=1.0)
        return averaged[theirs > 0]


@metric_registry.register("verification_accuracy")
class VerificationAccuracy(GalleryReading):
    """The share of the epoch's pairs that one cosine threshold, the best there is, calls right.

    The other question a direction answers. Ranking asks who a picture is nearest to and needs a gallery
    to answer at all; this asks whether two pictures are the same identity, which is what a door or a
    turnstile asks, and answers with a number that stands without one.

    Read against its baseline, which moves: with no threshold beating "these are all strangers", this
    reads the share of pairs that *are* strangers, which approaches one minus the reciprocal of however
    many identities the epoch holds — 0.80 over five, 0.97 over thirty. So an untrained model reads high
    here rather than near nothing, and two runs are comparable only over the same split. Recall carries
    the same caveat for the same reason, and it is spelled out where a run declares them.
    """

    @override
    def _read(self, embeddings: Tensor, identities: Tensor) -> Tensor:
        self._refuse_an_epoch_where_no_identity_repeats(identities)
        return _separated(self._blocks(embeddings, identities), identities).accuracy


@metric_registry.register("verification_threshold")
class VerificationThreshold(GalleryReading):
    """The cosine that separates the epoch's pairs best — the number a deployment compares against.

    Published beside the accuracy rather than inside it because it is what leaves the run: the accuracy
    says how well two pictures can be told apart, this says at what. One declaration answers with one
    number here, as everywhere in this framework, so the two are declared separately and each sweeps the
    epoch itself — what they share is the epoch, not the sweep of it. They cannot disagree all the same:
    the sweep is a function of state the two hold jointly, so the accuracy reported is the accuracy at
    the threshold reported. The price is the second sweep, measured at 5 ms over a thousand pictures and
    107 ms over five thousand, once per evaluation epoch and only where both are declared.
    """

    higher_is_better = None
    """Neither direction is better: shown with a best-so-far column, a threshold would name the epoch
    whose separation drifted furthest as the run's best."""

    @override
    def _read(self, embeddings: Tensor, identities: Tensor) -> Tensor:
        self._refuse_an_epoch_where_no_identity_repeats(identities)
        return _separated(self._blocks(embeddings, identities), identities).threshold


def _separated(blocks: Iterator[tuple[Tensor, Tensor, int]], identities: Tensor) -> Separation:
    """Sweep every threshold of the grid over one epoch's pairs and answer with the best of them.

    Counted rather than sorted: each pair is filed under the grid level below its cosine, which makes
    "filed at or above this level" the same statement as "at or above this threshold" and so makes what
    is reported exact — the accuracy given is the accuracy that very threshold gives, whatever the grid
    cost in finding it. Filing at the *nearest* level instead would report the accuracy of rounded
    cosines: measured over six galleries, up to 1.28e-03 away from what its own threshold delivers.

    What outlives a block is two histograms of a few thousand counters. Inside one, the filing and the
    mask of who matches whom are each the shape of the block of cosines itself, so while a block is
    alive this holds about three times what a ranking does — bounded by ``QUERY_BLOCK`` all the same,
    rather than by the epoch squared. Where several thresholds read alike the lowest is answered with,
    which is the field's convention and the only one the epoch holds a reason to prefer.
    """
    thresholds = torch.linspace(-1.0, 1.0, LEVELS, device=identities.device)
    matched = torch.zeros(LEVELS, dtype=torch.long, device=identities.device)
    apart = torch.zeros_like(matched)
    for similarity, mine, at in blocks:
        filed = torch.bucketize(similarity, thresholds, right=True, out_int32=True).sub_(1).clamp_(min=0)
        rows = torch.arange(len(mine), device=filed.device)
        # A picture is not a pair with itself: sent to the level past the end, which is sliced off below.
        filed[rows, rows + at] = LEVELS
        held = torch.bincount(filed.flatten(), minlength=LEVELS + 1)[:LEVELS]
        ours = torch.bincount(filed[identities == mine.unsqueeze(-1)], minlength=LEVELS + 1)[:LEVELS]
        matched += ours
        apart += held - ours
    found = matched.sum() - matched.cumsum(0) + matched
    rejected = apart.cumsum(0) - apart
    correct = found + rejected
    best = int((correct == correct.max()).nonzero()[0])
    return Separation(thresholds[best], correct[best] / int(matched.sum() + apart.sum()))
