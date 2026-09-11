"""A loss turns one task's raw output and its target into a number, under a name a report can show."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor

from src.config import ComponentConfig, WeightedLossConfig
from src.core import LossOutput, Semantics, TargetInfo
from src.losses import CrossEntropy, Expectation, Focal, Loss, TorchLoss, WeightedSum
from src.losses.build import build_loss
from src.losses.registry import loss_registry
from src.tasks import BinarySegmentation, Classification, Regression, Segmentation
from src.tasks.registry import task_registry
from tests.support.tasks import CLASSES as THREE
from tests.support.tasks import specimen as task_specimen

CLASSES = 3


NUMBERS = (torch.randn(4), torch.tensor([1.0, 2.0, 3.0, 4.0]))
SCORES = (torch.randn(4), torch.tensor([0.0, 1.0, 1.0, 0.0]))
REGIONS = (torch.randn(4, CLASSES, 6, 6), torch.zeros(4, 6, 6, dtype=torch.long))

SPECIMENS: dict[str, tuple[Tensor, Tensor]] = {
    "cross_entropy": (torch.randn(4, CLASSES), torch.tensor([0, 1, 2, 0])),
    # A binned target is a distribution over the bins, as its encoder wrote it.
    "expectation": (torch.randn(4, CLASSES), torch.eye(CLASSES)[[0, 1, 2, 0]]),
    "bce": SCORES,
    "focal": SCORES,
    "dice": REGIONS,
    "iou": REGIONS,
    "tversky": REGIONS,
    "mse": NUMBERS,
    "mae": NUMBERS,
    "huber": NUMBERS,
    "smooth_l1": NUMBERS,
}
"""Raw outputs and the target each family compares them with: a newly registered loss needs a row here."""


def specimen(name: str) -> tuple[Tensor, Tensor]:
    """A fresh pair each call: one test asks for gradients, and must not leave them on the next test's tensor."""
    if name not in SPECIMENS:
        raise LookupError(f"{name!r} is registered but has no specimen; add a row to SPECIMENS.")
    outputs, targets = SPECIMENS[name]
    return outputs.clone(), targets.clone()


BINS = TargetInfo(classes={0: "low", 1: "mid", 2: "high"}, values=(0.0, 1.0, 2.0))


REGIONAL = {"dice", "iou", "tversky"}
"""The losses that score a region, and so are the ones a task's label semantics reaches."""


def settled(info: TargetInfo | None = None, semantics: Semantics | None = None) -> dict[str, object]:
    """What a task answers about its target — the one table a loss is sized from."""
    settled = info if info is not None else TargetInfo()
    return {"semantics": semantics, "num_classes": settled.num_classes, "values": settled.values}


def facts(name: str) -> dict[str, object]:
    """The same table, for whichever registered loss is under test."""
    info = BINS if name == "expectation" else TargetInfo()
    return settled(info, Semantics.MULTICLASS if name in REGIONAL else None)


def built(name: str) -> Loss:
    """Each registered loss as its builder makes it: facts come from the task, never from a declaration."""
    return build_loss(ComponentConfig(name=name), facts(name))


class TestContract:
    @pytest.mark.parametrize("name", list(loss_registry))
    def test_every_registered_loss_reports_one_scalar_under_one_name(self, name: str) -> None:
        outputs, targets = specimen(name)

        result = built(name)(outputs, targets)

        assert isinstance(result, LossOutput) and result.total.ndim == 0
        assert list(result.losses) == [built(name).log_name]
        assert torch.equal(next(iter(result.losses.values())), result.total)

    @pytest.mark.parametrize("name", list(loss_registry))
    def test_every_registered_loss_carries_a_gradient_back_to_the_output(self, name: str) -> None:
        outputs, targets = specimen(name)
        outputs.requires_grad_(True)

        built(name)(outputs, targets).total.backward()

        assert outputs.grad is not None and torch.any(outputs.grad != 0)


class TestSemantics:
    """What a task's labels mean is settled once, by the task, and reaches whoever is sized from it."""

    def test_a_region_loss_is_told_what_the_labels_it_scores_mean(self) -> None:
        """`loss: dice` on a binary task built a multiclass loss silently and scored a degenerate one-hot."""
        for kind, mode in ((BinarySegmentation, "binary"), (Segmentation, "multiclass")):
            task = kind("mask", TargetInfo() if kind is BinarySegmentation else TargetInfo(classes=THREE))

            built = build_loss(ComponentConfig(name="dice"), task.facts())

            assert isinstance(built, TorchLoss) and built.module.mode == mode

    def test_a_region_loss_over_a_target_that_carries_no_labels_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no labels"):
            build_loss(ComponentConfig(name="dice"), Regression("age", TargetInfo()).facts())

    @pytest.mark.parametrize("kind", list(task_registry))
    def test_every_kinds_default_objective_scores_the_output_its_own_head_produces(self, kind: str) -> None:
        """The seam between a task and its loss: shapes, dtypes and the class axis agree with no help."""
        task, output, batch = task_specimen(kind)

        result = build_loss(task.default_loss, task.facts())(task.raw(output), task.loss_target(batch))

        assert result.total.ndim == 0 and bool(torch.isfinite(result.total))


class TestShapes:
    @pytest.mark.parametrize("name", ["bce", "mse", "mae"], ids=["binary", "squared error", "absolute error"])
    @pytest.mark.parametrize("shape", [(4,), (4, 1)], ids=["a plain number", "a width-one channel"])
    def test_a_single_output_channel_is_read_whether_or_not_the_head_kept_it(
        self, name: str, shape: tuple[int, ...]
    ) -> None:
        """A head produces `[B, 1]`; a target is `[B]`. Squeezing here is what keeps them comparable."""
        targets = torch.tensor([0.0, 1.0, 1.0, 0.0])

        result = built(name)(torch.rand(*shape), targets)

        assert result.total.ndim == 0

    def test_class_weights_are_declared_as_a_list_and_reach_the_loss_as_a_tensor(self) -> None:
        balanced = CrossEntropy()
        weighted = CrossEntropy(weight=[0.1, 0.1, 10.0])
        outputs, targets = torch.randn(4, CLASSES), torch.tensor([2, 2, 0, 0])

        assert weighted(outputs, targets).total != balanced(outputs, targets).total


class TestExpectation:
    def test_it_compares_the_numbers_two_distributions_stand_for(self) -> None:
        """A binned regression is learned as a distribution; this term keeps its mean on the wanted value."""
        loss = Expectation(values=(0.0, 10.0, 20.0))
        wanted = torch.tensor([[0.0, 1.0, 0.0]])

        exact = loss(torch.tensor([[-20.0, 20.0, -20.0]]), wanted)
        far = loss(torch.tensor([[20.0, -20.0, -20.0]]), wanted)

        assert exact.total < 0.1 and far.total == pytest.approx(10.0, abs=0.5)

    def test_an_output_that_does_not_span_the_bins_is_refused_by_count(self) -> None:
        loss = Expectation(values=(0.0, 10.0))

        with pytest.raises(ValueError, match="2 bins"):
            loss(torch.randn(4, 3), torch.rand(4, 2))


class TestWeightedSum:
    @pytest.fixture
    def parts(self) -> list[tuple[Loss, float]]:
        return [(built("cross_entropy"), 1.0), (built("dice"), 0.5)]

    def test_it_sums_weighted_totals_while_reporting_each_loss_unweighted(
        self, parts: list[tuple[Loss, float]]
    ) -> None:
        """A weight changes the objective, never the number a report shows: `ce` means `ce`."""
        outputs, targets = specimen("dice")
        alone = [loss(outputs, targets).total for loss, _ in parts]

        result = WeightedSum(parts)(outputs, targets)

        assert result.total == pytest.approx(float(alone[0] + 0.5 * alone[1]), abs=1e-5)
        assert result.losses["cross_entropy"] == pytest.approx(float(alone[0]), abs=1e-5)
        assert result.contributions["dice"] == pytest.approx(float(0.5 * alone[1]), abs=1e-5)

    def test_two_terms_that_would_log_under_one_name_are_refused(self) -> None:
        summed = WeightedSum([(built("dice"), 1.0), (built("dice"), 1.0)])

        with pytest.raises(ValueError, match="dice"):
            summed(*specimen("dice"))

    def test_a_sum_of_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            WeightedSum([])


class TestBuild:
    """A task's objective comes from its declaration, or from the default its own semantics names."""

    def test_a_task_default_is_written_the_same_way_a_run_writes_one(self) -> None:
        """A kind declares its objective as a name; nothing special happens to it on the way here."""
        task = Classification("species", TargetInfo(classes={0: "cat", 1: "dog"}))

        assert isinstance(build_loss(task.default_loss, task.facts()), CrossEntropy)

    def test_one_declared_component_becomes_that_loss(self) -> None:
        assert isinstance(build_loss(ComponentConfig(name="focal"), settled()), Focal)

    def test_several_declared_losses_become_their_weighted_sum(self) -> None:
        declared = [
            WeightedLossConfig(loss=ComponentConfig(name="cross_entropy"), weight=1.0),
            WeightedLossConfig(loss=ComponentConfig(name="dice"), weight=0.5),
        ]

        built = build_loss(declared, settled(TargetInfo(classes={0: "a", 1: "b"}), Semantics.MULTICLASS))

        assert isinstance(built, WeightedSum) and built.weights == [1.0, 0.5]

    def test_a_binned_regression_gets_the_pair_its_encoder_implies(self) -> None:
        """The bins come from the encoder, and with them the objective that reads them; nothing is written."""
        binned = TargetInfo(classes={0: "a", 1: "b", 2: "c"}, values=(0.0, 5.0, 10.0))
        task = Regression("age", binned)

        built = build_loss(task.default_loss, task.facts())

        assert isinstance(built, WeightedSum) and built.weights == [1.0, 0.5]
        assert isinstance(built.parts[1], Expectation) and built.parts[1].values.tolist() == [0.0, 5.0, 10.0]

    def test_a_fact_the_encoder_already_stated_is_refused_where_it_was_restated(self) -> None:
        binned = TargetInfo(classes={0: "a", 1: "b"}, values=(0.0, 1.0))
        declared = ComponentConfig.model_validate({"name": "expectation", "values": [0.0, 2.0]})

        with pytest.raises(ValueError, match="values"):
            build_loss(declared, settled(binned))

    def test_a_loss_reached_by_import_path_is_wrapped_under_the_name_it_is_declared_with(self) -> None:
        """Any torch loss is usable without a class of ours; the log name is the one thing we have to add."""
        declared = ComponentConfig.model_validate({"_target_": "torch.nn.SmoothL1Loss", "beta": 0.5})

        built = build_loss(declared, settled())

        assert built(torch.rand(4), torch.rand(4)).total.ndim == 0
        assert list(built(torch.rand(4), torch.rand(4)).losses) == ["smooth_l1"]

    def test_a_term_reports_under_the_name_the_run_wrote_for_it(self) -> None:
        """`loss: dice` reads back as `dice` in the metrics; `log_name` tells two terms of a kind apart."""
        declared = [
            WeightedLossConfig(loss=ComponentConfig(name="dice"), log_name="region"),
            WeightedLossConfig(loss=ComponentConfig(name="cross_entropy"), weight=0.5),
        ]

        built = build_loss(declared, settled(TargetInfo(classes={0: "a", 1: "b"}), Semantics.MULTICLASS))

        reported = built(torch.randn(2, 2, 4, 4), torch.zeros(2, 4, 4, dtype=torch.long))
        assert sorted(reported.losses) == ["cross_entropy", "region"]

    def test_a_declaration_that_is_not_a_loss_at_all_is_named(self) -> None:
        declared = ComponentConfig.model_validate({"_target_": "torch.nn.Linear", "in_features": 2, "out_features": 2})

        with pytest.raises(TypeError, match="Linear"):
            build_loss(declared, settled())


class TestDeclaration:
    """Nothing about a loss is written twice: its arguments are the library's, its name is the run's."""

    @pytest.mark.parametrize(
        ("declared", "expected"),
        [
            pytest.param({"name": "cross_entropy"}, "cross_entropy", id="the name a run wrote"),
            pytest.param({"name": "cross_entropy", "label_smoothing": 0.1}, "cross_entropy", id="with arguments"),
            pytest.param({"_target_": "torch.nn.L1Loss"}, "l1", id="a torch loss by import path"),
        ],
    )
    def test_a_term_is_reported_under_the_name_it_was_declared_with(
        self, declared: dict[str, Any], expected: str
    ) -> None:
        built = build_loss(ComponentConfig.model_validate(declared), settled())

        assert built.log_name == expected

    @pytest.mark.parametrize(
        ("name", "option", "value"),
        [
            pytest.param("cross_entropy", "label_smoothing", 0.1, id="a number torch accepts"),
            pytest.param("cross_entropy", "weight", [1.0, 2.0, 4.0], id="per-class weights, written as a list"),
            pytest.param("dice", "smooth", 0.5, id="a number smp accepts"),
            pytest.param("dice", "classes", [0, 2], id="a list smp wants as a list"),
            pytest.param("huber", "delta", 0.5, id="an argument no loss of ours mentions"),
        ],
    )
    def test_every_argument_the_library_takes_reaches_it_without_being_listed_here(
        self, name: str, option: str, value: Any
    ) -> None:
        """A loss of ours declares no argument list: whatever the library documents, a run may write."""
        built = build_loss(ComponentConfig.model_validate({"name": name, option: value}), facts(name))

        assert isinstance(built, TorchLoss)
        held = getattr(built.module, option)
        assert held.tolist() == value if isinstance(held, Tensor) else held == value
