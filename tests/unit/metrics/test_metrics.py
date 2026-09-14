"""What a run is judged by: the names a declaration writes, and the one reading that means an image."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from torch import Tensor
from torchmetrics.retrieval import RetrievalHitRate, RetrievalMAP

from src.core import Matrix, Semantics, Stage
from src.metrics import ConfusionMatrix, GalleryReading
from src.metrics.build import build_metrics
from src.metrics.metric_learning import (
    QUERY_BLOCK,
    MeanAveragePrecision,
    RecallAtK,
    VerificationAccuracy,
    VerificationThreshold,
)
from src.metrics.registry import metric_registry
from src.tasks.registry import task_registry
from tests.support.tasks import published, specimen

CLASSES = 3
VOTES = torch.rand(4, CLASSES).softmax(dim=1)
CHOICES = torch.tensor([0, 1, 2, 0])
PAIRED = torch.tensor([0, 0, 1, 1])
"""Two identities, two pictures each: what a reading averaged over every match a query has needs."""
NUMBERS = torch.tensor([1.5, 2.5, 3.5, 4.5])
VOCABULARY: Mapping[str, object] = {"semantics": Semantics.MULTICLASS, "num_classes": CLASSES}

SPECIMENS: dict[str, tuple[Mapping[str, object], Tensor, Tensor]] = {
    "accuracy": (VOCABULARY, VOTES, CHOICES),
    "f1": (VOCABULARY, VOTES, CHOICES),
    "precision": (VOCABULARY, VOTES, CHOICES),
    "recall": (VOCABULARY, VOTES, CHOICES),
    "confusion_matrix": (VOCABULARY, VOTES, CHOICES),
    "iou": (VOCABULARY, torch.rand(4, CLASSES, 6, 6), torch.zeros(4, 6, 6, dtype=torch.long)),
    "mae": ({}, NUMBERS, NUMBERS + 1),
    "mse": ({}, NUMBERS, NUMBERS + 1),
    # Directions, and which identity each of four samples is: what a retrieval reading ranks.
    "recall_at_k": ({}, torch.eye(4), CHOICES),
    "map": ({}, torch.eye(4), PAIRED),
    "verification_accuracy": ({}, torch.eye(4), PAIRED),
    "verification_threshold": ({}, torch.eye(4), PAIRED),
}
"""The facts each name is sized by and a batch it scores: a newly registered name needs a row here."""


class TestContract:
    @pytest.mark.parametrize("name", list(metric_registry))
    def test_every_registered_name_scores_the_batch_its_facts_describe(self, name: str) -> None:
        assert name in SPECIMENS, f"{name!r} is registered but has no specimen batch; add one to SPECIMENS."
        facts, predictions, targets = SPECIMENS[name]

        metrics = build_metrics({name: {"name": name}}, facts)
        metrics.update(predictions, targets)
        computed = metrics.compute()

        assert set(computed) == {name} and isinstance(computed[name], Tensor | Matrix)

    @pytest.mark.parametrize("kind", list(task_registry))
    def test_the_metrics_a_kind_declares_score_what_that_kind_predicts(self, kind: str) -> None:
        """A task's table and the views it answers with have to fit each other, whatever its semantics."""
        task, output, batch = specimen(kind)
        declared = type(task).default_metrics

        metrics = build_metrics(declared, task.facts())
        metrics.update(published(task, output), task.metric_view(batch))

        assert set(metrics.compute()) == set(declared)

    def test_two_flavours_of_one_metric_stand_side_by_side(self) -> None:
        """The label is what a report shows, so it is the declaration's to choose, not the metric's."""
        metrics = build_metrics(
            {"f1": {"name": "f1", "average": "macro"}, "f1_per_class": {"name": "f1", "average": "none"}}, VOCABULARY
        )
        metrics.update(VOTES, CHOICES)

        computed = metrics.compute()

        assert computed["f1"].ndim == 0 and computed["f1_per_class"].shape == (CLASSES,)


class TestConfusionMatrix:
    def test_it_says_which_of_its_axes_is_the_prediction(self) -> None:
        """A bare square tensor is any two-dimensional reading; which axis is which is this class's to say."""
        matrix = ConfusionMatrix(semantics=Semantics.MULTICLASS, num_classes=CLASSES)
        matrix.update(VOTES, CHOICES)

        drawn = matrix.compute()

        assert isinstance(drawn, Matrix) and drawn.value.shape == (CLASSES, CLASSES)
        assert (drawn.xaxis, drawn.yaxis) == ("Predicted", "True")

    def test_it_forgets_its_counts_with_the_metric_that_holds_them(self) -> None:
        """Measured on torchmetrics 1.9.0: resetting a metric leaves the state of a metric inside it."""
        matrix = ConfusionMatrix(semantics=Semantics.MULTICLASS, num_classes=CLASSES)
        matrix.update(VOTES, CHOICES)

        matrix.reset()
        matrix.update(VOTES, CHOICES)

        assert matrix.compute().value.sum() == len(CHOICES)

    def test_it_is_never_folded_into_another_metrics_compute_group(self) -> None:
        """Why it holds its counter instead of inheriting one: a metric with no state of its own is never
        grouped (torchmetrics 1.9.0). Were that to change, a collection would update only the group's
        leader and this matrix would draw stale counts — a wrong chart, not an error."""
        grouped = build_metrics({"f1": {"name": "f1"}, "confusion_matrix": {"name": "confusion_matrix"}}, VOCABULARY)
        alone = ConfusionMatrix(semantics=Semantics.MULTICLASS, num_classes=CLASSES)

        for _ in range(2):
            grouped.update(VOTES, CHOICES)
            alone.update(VOTES, CHOICES)

        assert torch.equal(grouped.compute()["confusion_matrix"].value, alone.compute().value)

    def test_a_multilabel_matrix_is_refused_where_it_was_declared(self) -> None:
        """One small matrix per label draws as nothing; saying so before the run beats logging silence."""
        with pytest.raises(ValueError, match="multilabel"):
            build_metrics(
                {"confusion_matrix": "confusion_matrix"},
                {"semantics": Semantics.MULTILABEL, "num_classes": CLASSES},
            )


GALLERY_READINGS: list[type[GalleryReading]] = [
    RecallAtK,
    MeanAveragePrecision,
    VerificationAccuracy,
    VerificationThreshold,
]
"""Every reading taken by comparing one epoch's directions with each other: a new one is a row here."""

NEEDS_A_REPEATED_IDENTITY: list[type[GalleryReading]] = [
    MeanAveragePrecision,
    VerificationAccuracy,
    VerificationThreshold,
]
"""Those of them an epoch of strangers says nothing about; recall reads its floor there, not its ceiling."""


class TestGalleryReadings:
    """What every reading taken from a whole epoch owes, whatever question it then asks of it."""

    @pytest.mark.parametrize("reading", GALLERY_READINGS)
    def test_a_gallery_holding_one_identity_is_refused_rather_than_read_as_a_perfect_score(
        self, reading: type[GalleryReading]
    ) -> None:
        """Every neighbour is then a match, so the reading is its own ceiling whatever the model learned.

        Measured on an untrained encoder over 148 samples: recall@1 reads 1.0000 over one identity,
        0.5878 over two and 0.0541 over 37. One identity is the case that is degenerate by construction,
        and the only one a floor can be drawn at without choosing a number nobody can defend.
        """
        metric = reading()
        metric.update(torch.randn(6, 8), torch.zeros(6, dtype=torch.long))

        with pytest.raises(ValueError, match="one identity"):
            metric.compute()

    @pytest.mark.parametrize("reading", NEEDS_A_REPEATED_IDENTITY)
    def test_an_epoch_where_no_identity_repeats_is_refused_rather_than_answered(
        self, reading: type[GalleryReading]
    ) -> None:
        """With nobody to find, a mean over all of them is a mean over none and a threshold rejects all.

        Recall is left out of this table deliberately: it reads 0.0 there, which is its floor rather than
        its ceiling, and a reading that cannot succeed says something true about a split of strangers.
        """
        metric = reading()
        metric.update(torch.eye(4), torch.tensor([0, 1, 2, 3]))

        with pytest.raises(ValueError, match="another of its own identity"):
            metric.compute()

    @pytest.mark.parametrize("reading", GALLERY_READINGS)
    def test_a_gallery_holding_a_direction_that_is_not_one_is_refused_rather_than_ranked(
        self, reading: type[GalleryReading]
    ) -> None:
        """A number that is not a number ranks, compares and files like any other, and reads as a score.

        Measured on four samples of two identities with a single NaN among the thirty-two values:
        recall@1 answered 0.5, mAP 0.6667 and verification 0.5 — a plausible epoch from an encoder that
        had produced nothing of the sort. Refused on the gallery rather than in each reading, because
        every one of them loses the evidence at the same step: after `topk`, after a comparison, after
        `bucketize`, a NaN is an ordinary index.
        """
        embeddings = torch.eye(4)
        embeddings[0, 0] = float("nan")
        metric = reading()
        metric.update(embeddings, torch.tensor([0, 0, 1, 1]))

        with pytest.raises(ValueError, match="1 of 4"):
            metric.compute()

    def test_every_reading_of_one_epoch_shares_the_one_copy_of_it(self) -> None:
        """Measured on torchmetrics 1.9.0: metrics whose state agrees after a batch become one compute
        group, after which only the group's leader is updated and the rest are handed its state by
        reference. So a run declaring four gallery readings holds one epoch of directions, not four.
        Were that to change, memory would quietly multiply by however many readings were declared."""
        collection = build_metrics({name.lower(): {"name": name.lower()} for name in ["map", "recall_at_k"]}, {})
        collection.update(torch.eye(4), PAIRED)
        collection.update(torch.eye(4), PAIRED)

        leader, *rest = next(iter(collection._groups.values()))
        held = [getattr(collection, one).embeddings for one in rest]
        assert held and all(one is getattr(collection, leader).embeddings for one in held)

    @pytest.mark.parametrize("reading", GALLERY_READINGS)
    def test_every_reading_of_a_gallery_is_taken_where_the_encoder_is_standing_still(
        self, reading: type[GalleryReading]
    ) -> None:
        """A gallery accumulated while the encoder still moves holds vectors from several models."""
        assert reading.read_on == frozenset({Stage.VAL, Stage.TEST})


class TestRetrieval:
    """Ranking against a gallery: the reading whose answer needs more of an epoch than one batch holds."""

    def test_recall_is_read_from_the_whole_epoch_rather_than_from_one_batch(self) -> None:
        """The gallery a query is ranked against is the epoch; a batch of its own is not one."""
        directions = torch.eye(3)
        identities = torch.tensor([0, 1, 2])
        metric = RecallAtK(k=1)

        metric.update(directions, identities)
        metric.update(directions, identities)

        assert float(metric.compute()) == 1.0, "each sample's only match is the twin in the other batch"

    def test_average_precision_is_read_from_the_whole_epoch_rather_than_from_one_batch(self) -> None:
        """Within either batch alone no two pictures share an identity, so neither could be ranked at all."""
        directions = torch.eye(3)
        identities = torch.tensor([0, 1, 2])
        metric = MeanAveragePrecision()

        metric.update(directions, identities)
        metric.update(directions, identities)

        assert float(metric.compute()) == 1.0, "each picture's only match is its twin in the other batch"

    def test_it_reads_what_the_library_reads_for_the_same_epoch(self) -> None:
        """torchmetrics is this metric's specification; the matrix below is only a cheaper way to it.

        ``RetrievalHitRate`` wants one row per (query, candidate) pair, which a gallery ranked against
        itself has to be unrolled into. Agreeing with it here is what lets the ranking done in blocks
        below be trusted. Ties are deliberately not part of the comparison: where two neighbours score
        alike the two implementations may pick different ones, and both readings are equally right.
        """
        identities = torch.randint(0, 8, (60,))
        embeddings = torch.nn.functional.normalize(torch.randn(60, 16), dim=1)
        pairs = ~torch.eye(len(identities), dtype=torch.bool)
        similarity = embeddings @ embeddings.T

        for k in (1, 5):
            ours = RecallAtK(k=k)
            ours.update(embeddings, identities)
            library = RetrievalHitRate(top_k=k)(
                similarity[pairs],
                (identities.unsqueeze(1) == identities.unsqueeze(0))[pairs],
                indexes=torch.arange(len(identities)).unsqueeze(1).expand_as(similarity)[pairs],
            )
            assert float(ours.compute()) == pytest.approx(float(library)), f"at k={k}"

    def test_mean_average_precision_reads_what_the_library_reads_for_the_same_epoch(self) -> None:
        """Recall asks whether the first neighbour was one of yours; this asks about all of them.

        The reading to watch where an identity has many pictures: a model that finds one of six and
        buries the rest reads the same as one that finds all six, under recall, and differently here.

        The library is scored on the same ranking moved into positive numbers, and that is not a fudge
        but the only way it can answer about cosines at all. Measured on torchmetrics 1.9.0:
        ``retrieval_average_precision`` opens with ``target = torch.where(preds > 0, target, 0)``, so a
        match whose score is at or below zero is silently dropped — with half a cosine's range below
        zero, it read 0.1429 where the average precision of those very ranks is 0.0848. Average
        precision is a function of the ranking alone, so a shift that preserves every order changes
        nothing about the question and everything about whether the library may be asked it.

        Two things about the epoch below are the fixture rather than decoration, and this test failed
        for both before they were. Every identity appears more than once, because a query with nobody of
        its own is the one case where the two readings differ *by design* — this one leaves it out and
        the library scores it nought, which is what the test below this states — and a drawn identity
        was lonely in 7 of 200 draws, every one of them a disagreement. And the directions are drawn in
        double, because two near-equal cosines can sort one way here and the other way in the library:
        measured over 200 draws, in single they part by up to 8.1e-06 — above any tolerance at which
        this test would still be saying anything — and in double by 3.0e-08.
        """
        identities = torch.arange(8).repeat(8)
        embeddings = torch.nn.functional.normalize(torch.randn(64, 16, dtype=torch.float64), dim=1)
        pairs = ~torch.eye(len(identities), dtype=torch.bool)
        similarity = embeddings @ embeddings.T
        ours = MeanAveragePrecision()

        ours.update(embeddings, identities)

        library = RetrievalMAP()(
            similarity[pairs] + 2.0,
            (identities.unsqueeze(1) == identities.unsqueeze(0))[pairs],
            indexes=torch.arange(len(identities)).unsqueeze(1).expand_as(similarity)[pairs],
        )
        assert float(ours.compute()) == pytest.approx(float(library), abs=1e-6)

    def test_a_query_with_nobody_of_its_own_in_the_epoch_is_left_out_rather_than_scored_zero(self) -> None:
        """Its average precision is undefined, not nought: there was nothing for it to rank well.

        Scored zero, the reading would fall as a split held more one-off identities — a number about how
        the data was cut rather than about the model.
        """
        directions = torch.eye(5)
        twins_then_singles = torch.tensor([0, 0, 1, 2, 3])
        metric = MeanAveragePrecision()

        metric.update(directions, twins_then_singles)

        assert float(metric.compute()) == 1.0, "the one query with a match found it; the other three had none"

    def test_an_epoch_wider_than_one_block_is_read_as_one_gallery(self) -> None:
        """Queries are ranked a block at a time; a sample's own column then sits where its block starts.

        Getting that offset wrong excludes somebody else's sample from every block but the first, which
        changes the reading by a little and looks like nothing.
        """
        held = QUERY_BLOCK * 2 + 37
        identities = torch.arange(held) % 50
        embeddings = torch.nn.functional.normalize(torch.randn(held, 8), dim=1)
        metric = RecallAtK(k=3)

        metric.update(embeddings, identities)

        pairs = ~torch.eye(held, dtype=torch.bool)
        similarity = embeddings @ embeddings.T
        library = RetrievalHitRate(top_k=3)(
            similarity[pairs],
            (identities.unsqueeze(1) == identities.unsqueeze(0))[pairs],
            indexes=torch.arange(held).unsqueeze(1).expand_as(similarity)[pairs],
        )
        assert float(metric.compute()) == pytest.approx(float(library))

    def test_it_ranks_by_angle_however_loud_the_model_answers(self) -> None:
        """Left as dot products the ranking follows length, and a long vector becomes everyone's neighbour.

        A task publishes unit vectors, so the shipped path never shows this; a reading pointed at raw
        outputs would silently answer about magnitudes instead of directions.
        """
        directions = torch.nn.functional.normalize(torch.randn(40, 8), dim=1)
        identities = torch.randint(0, 5, (40,))
        loud = directions * torch.rand(40, 1).add(0.1).mul(50)

        quiet, shouted = RecallAtK(k=3), RecallAtK(k=3)
        quiet.update(directions, identities)
        shouted.update(loud, identities)

        assert float(shouted.compute()) == pytest.approx(float(quiet.compute()))

    def test_a_sample_is_never_its_own_nearest_neighbour(self) -> None:
        """Left in the ranking, every query retrieves itself and an untrained model reads a perfect score.

        Six samples, six identities, and each direction repeated once: the only neighbour any of them
        has is somebody else, so nothing here can be found and the reading is zero.
        """
        directions = torch.eye(3)
        metric = RecallAtK(k=1)

        metric.update(directions, torch.tensor([0, 1, 2]))
        metric.update(directions, torch.tensor([3, 4, 5]))

        assert float(metric.compute()) == 0.0

    def test_an_epoch_with_nobody_to_rank_against_is_refused_by_name(self) -> None:
        """`recall_at_10` over a split of eight is a declaration no epoch can answer; say which two disagree."""
        metric = RecallAtK(k=4)

        metric.update(torch.eye(3), torch.tensor([0, 1, 2]))

        with pytest.raises(ValueError, match="4"):
            metric.compute()

    def test_a_ranking_that_looks_nowhere_is_refused_where_it_is_declared(self) -> None:
        with pytest.raises(ValueError, match="'k'"):
            RecallAtK(k=0)

    def test_a_reading_that_is_better_higher_says_so_for_whoever_draws_it(self) -> None:
        assert RecallAtK(k=1).higher_is_better is True


def separating(pictures: int, identities: int, *, spread: float) -> tuple[Tensor, Tensor]:
    """A gallery a model has learned something about: every identity around a direction of its own.

    Random directions will not do for a reading about thresholds. With several identities and nothing
    learned, no threshold beats calling every pair a stranger — measured, a gallery of fifty over six
    identities read 0.8204, which is exactly the share of pairs that are strangers — and a reading whose
    answer is that is never asked anything about where it drew its line. Built without a generator, so
    what it looks like does not depend on how many draws the tests before it happened to take.
    """
    who = torch.arange(pictures) % identities
    jitter = torch.sin(torch.arange(pictures * identities, dtype=torch.float32)).reshape(pictures, identities)
    return torch.eye(identities)[who] + spread * jitter, who


def accuracy_at(embeddings: Tensor, identities: Tensor, threshold: float) -> float:
    """The plain sentence, counted directly: of every pair of pictures, how many this threshold calls right.

    The oracle the reading is scored against — written out rather than derived, because what it asserts
    is that a number reported under this name means exactly this and not a version of it.
    """
    directions = torch.nn.functional.normalize(embeddings, dim=1)
    pairs = ~torch.eye(len(identities), dtype=torch.bool)
    same = (identities.unsqueeze(0) == identities.unsqueeze(1))[pairs]
    said_same = (directions @ directions.T)[pairs] >= threshold
    return float((said_same == same).float().mean())


def best_over_every_observed_value(embeddings: Tensor, identities: Tensor) -> float:
    """The best accuracy any threshold at all can reach, found by trying every value the epoch holds.

    Accuracy only steps where a pair sits, so the values themselves are the whole candidate set. Too
    expensive to run over a real epoch — it is quadratic in the pairs — and exactly right over a small one.
    """
    directions = torch.nn.functional.normalize(embeddings, dim=1)
    pairs = ~torch.eye(len(identities), dtype=torch.bool)
    similarity = (directions @ directions.T)[pairs]
    tried = torch.cat([similarity.unique(), similarity.max().add(1.0).unsqueeze(0)])
    return max(accuracy_at(embeddings, identities, float(one)) for one in tried)


class TestVerification:
    def test_rejecting_every_pair_is_a_decision_this_sweep_can_reach(self) -> None:
        """A collapsed encoder answers every pair alike, and then the only useful decision is to accept none.

        Measured before this held: four identical directions over two identities read an accuracy of
        1/3, while rejecting every pair — the baseline anybody reaches without a model — gives 2/3. The
        grid ended at 1.0 and the rule is `cosine >= threshold`, so a pair sitting at exactly 1.0 could
        not be refused by any candidate on it. A reading below its own trivial baseline is not a reading.
        """
        collapsed = torch.tensor([[1.0, 0.0]]).repeat(4, 1)
        identities = torch.tensor([0, 0, 1, 1])
        accuracy, threshold = VerificationAccuracy(), VerificationThreshold()
        accuracy.update(collapsed, identities)
        threshold.update(collapsed, identities)

        assert float(accuracy.compute()) == pytest.approx(4 / 6)
        assert float(threshold.compute()) > 1.0

    """Telling two pictures of one identity from two of different ones: one threshold over the cosines."""

    def test_the_accuracy_it_reports_is_the_accuracy_at_the_threshold_it_reports(self) -> None:
        """The two are read back as a pair — deploy this number, get that accuracy — so they cannot drift.

        Stated here over an ordinary gallery, and again below over the one case that can break it: where
        a best threshold sits on a plateau, which is most of the time, filing every pair a level away
        moves the threshold reported and leaves the pair agreeing with itself all the same.
        """
        embeddings, identities = separating(120, 6, spread=0.35)
        accuracy, threshold = VerificationAccuracy(), VerificationThreshold()

        for metric in (accuracy, threshold):
            metric.update(embeddings, identities)

        at = float(threshold.compute())
        assert float(accuracy.compute()) == pytest.approx(accuracy_at(embeddings, identities, at), abs=1e-9)

    def test_the_pair_it_reports_stays_honest_where_a_single_grid_step_decides(self) -> None:
        """The case the filing has to get exactly right, and close to the only one in which it shows.

        Accuracy over thresholds is a staircase and its best is usually a plateau; on one, filing every
        pair a level away moves the threshold reported without moving the accuracy, and the two go on
        agreeing. Here the best is one level wide by construction — three pictures, a twin pair just
        above nought and a stranger just below it — so the level between them calls all three pairs
        right and either of its neighbours calls one of them wrong.
        """
        directions = torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.0, 0.0], [0.0005, 1.0, 0.0], [-0.0005, -0.8, 0.6]]), dim=1
        )
        identities = torch.tensor([0, 0, 1])
        accuracy, threshold = VerificationAccuracy(), VerificationThreshold()

        for metric in (accuracy, threshold):
            metric.update(directions, identities)

        assert float(accuracy.compute()) == 1.0
        assert accuracy_at(directions, identities, float(threshold.compute())) == 1.0

    def test_its_grid_of_thresholds_is_fine_enough_to_find_what_an_exact_sweep_finds(self) -> None:
        """Only *which* threshold is grid-bound; measured against an exact sweep over six galleries, a
        grid of 2001 levels gave up at most 2.23e-05 of accuracy, and one of 201 up to 1.34e-04."""
        embeddings, identities = separating(120, 6, spread=0.35)
        metric = VerificationAccuracy()

        metric.update(embeddings, identities)

        assert float(metric.compute()) >= best_over_every_observed_value(embeddings, identities) - 1e-3

    def test_a_gallery_that_separates_cleanly_is_read_as_separating_cleanly(self) -> None:
        """Two identities, two pictures each, every picture pointing exactly where its twin does."""
        directions = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        metric = VerificationAccuracy()

        metric.update(directions, PAIRED)

        assert float(metric.compute()) == 1.0

    def test_where_several_thresholds_read_alike_the_lowest_of_them_is_the_one_reported(self) -> None:
        """The number leaves the run and is compared against in production, so which of a plateau it is
        cannot depend on how a library breaks a tie. Twins pointing exactly together and strangers
        exactly apart: every threshold above nought and up to one calls all twelve pairs right, and the
        lowest of those is the first level of the grid past nought."""
        directions = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        metric = VerificationThreshold()

        metric.update(directions, PAIRED)

        assert float(metric.compute()) == pytest.approx(0.001, abs=1e-6), "the grid is spread in float32"

    def test_a_picture_is_never_a_pair_with_itself(self) -> None:
        """Left in, every picture pairs with itself at a cosine of one and is always called right.

        Three pictures: two of one identity a quarter turn apart, and a stranger sitting between them, so
        no threshold can do better than rejecting everything — four of the six pairs. Counting the three
        self-pairs would make it seven of nine, a reading of the arithmetic rather than of the model.
        """
        half = 2.0**-0.5
        directions = torch.tensor([[1.0, 0.0], [0.0, 1.0], [half, half]])
        metric = VerificationAccuracy()

        metric.update(directions, torch.tensor([0, 0, 1]))

        assert float(metric.compute()) == pytest.approx(4 / 6)

    def test_it_is_read_from_the_whole_epoch_rather_than_from_one_batch(self) -> None:
        """Within either batch alone no two pictures share an identity, so neither can be read at all."""
        directions = torch.eye(3)
        metric = VerificationAccuracy()

        metric.update(directions, torch.tensor([0, 1, 2]))
        metric.update(directions, torch.tensor([0, 1, 2]))

        assert float(metric.compute()) == 1.0, "each picture's only match is its twin in the other batch"

    def test_an_epoch_wider_than_one_block_is_read_as_one_gallery(self) -> None:
        """Pairs are counted a block at a time, and a picture's own column sits where its block starts.

        Getting that offset wrong leaves somebody else's pair out of every block but the first and
        keeps a picture paired with itself instead, which moves the reading by a little and looks like
        nothing. The identities are laid out so that the pair wrongly dropped is a pair of *twins*:
        with strangers on both sides of the swap the two mistakes cancel to within a pair or two, and
        the reading comes back right for the wrong reason.
        """
        embeddings, identities = separating(QUERY_BLOCK + 17, 8, spread=0.45)
        metric, threshold = VerificationAccuracy(), VerificationThreshold()

        for one in (metric, threshold):
            one.update(embeddings, identities)

        assert float(metric.compute()) == pytest.approx(
            accuracy_at(embeddings, identities, float(threshold.compute())), abs=1e-9
        )

    def test_it_reads_angles_however_loud_the_model_answers(self) -> None:
        """A threshold over dot products would follow vector length; over cosines it follows direction."""
        directions = torch.nn.functional.normalize(torch.randn(40, 8), dim=1)
        identities = torch.randint(0, 5, (40,))
        loud = directions * torch.rand(40, 1).add(0.1).mul(50)

        quiet, shouted = VerificationAccuracy(), VerificationAccuracy()
        quiet.update(directions, identities)
        shouted.update(loud, identities)

        assert float(shouted.compute()) == pytest.approx(float(quiet.compute()))

    def test_a_threshold_is_neither_better_high_nor_low_and_says_so(self) -> None:
        """A cosine to compare against is not a score: shown with a best-so-far column it would name the
        epoch whose threshold drifted furthest as the run's best."""
        assert VerificationThreshold().higher_is_better is None
        assert VerificationAccuracy().higher_is_better is True
