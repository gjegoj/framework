"""What a run is judged by: the names a declaration writes, and the one reading that means an image."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from torch import Tensor
from torchmetrics.retrieval import RetrievalHitRate

from src.core import Matrix, Semantics, require_tensor
from src.metrics import ConfusionMatrix
from src.metrics.build import build_metrics
from src.metrics.metric_learning import QUERY_BLOCK, RecallAtK
from src.metrics.registry import metric_registry
from src.tasks.registry import task_registry
from tests.support.tasks import specimen

CLASSES = 3
VOTES = torch.rand(4, CLASSES).softmax(dim=1)
CHOICES = torch.tensor([0, 1, 2, 0])
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
        metrics.update(require_tensor(task.postprocess(output), name=task.name), task.metric_view(batch))

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
