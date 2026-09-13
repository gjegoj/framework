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
WIDTH = 5
IDENTITIES = torch.tensor([0, 1, 2, 0])
"""An embedding's width, and which identity each of four samples is — what an angular loss compares."""

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
    # Cosines against prototypes a head already holds, and the raw embeddings a loss holds them for.
    "arcface": (torch.rand(4, CLASSES) * 2 - 1, IDENTITIES),
    "arcface_proxy": (torch.randn(4, WIDTH), IDENTITIES),
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

ANGULAR = {"arcface", "arcface_proxy"}
"""The losses over an identity, which are sized by how many there are and how wide an embedding is."""


def settled(info: TargetInfo | None = None, semantics: Semantics | None = None) -> dict[str, object]:
    """What a task answers about its target — the one table a loss is sized from."""
    settled = info if info is not None else TargetInfo()
    return {"semantics": semantics, "num_classes": settled.num_classes, "values": settled.values}


def facts(name: str) -> dict[str, object]:
    """The same table, for whichever registered loss is under test."""
    info = BINS if name == "expectation" else TargetInfo()
    if name in ANGULAR:
        return {**settled(TargetInfo(classes=THREE)), "embedding_dim": WIDTH}
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


class TestAngular:
    """An angular margin: the prototypes it compares against, and the penalty that is the whole point."""

    @pytest.mark.parametrize("name", sorted(ANGULAR))
    def test_a_margin_makes_the_identity_a_sample_already_is_harder_to_keep(self, name: str) -> None:
        """Without this the loss is ordinary cross-entropy wearing the name of one that separates."""
        outputs, targets = specimen(name)

        with_margin = build_loss(ComponentConfig(name=name, margin=0.5), facts(name))
        flat = build_loss(ComponentConfig(name=name, margin=0.0), facts(name))
        flat.load_state_dict(with_margin.state_dict())

        assert float(with_margin(outputs, targets).total) > float(flat(outputs, targets).total)

    def test_the_prototypes_a_proxy_objective_compares_against_are_learned_with_the_run(self) -> None:
        """One per identity, held by the objective — so what a run ships carries none of them."""
        outputs, targets = specimen("arcface_proxy")

        built = build_loss(ComponentConfig(name="arcface_proxy"), facts("arcface_proxy"))
        built(outputs, targets).total.backward()

        prototypes = dict(built.named_parameters())["prototypes"]
        assert prototypes.shape == (len(THREE), WIDTH)
        assert prototypes.grad is not None and torch.any(prototypes.grad != 0)

    def test_an_objective_reading_a_head_that_already_holds_them_carries_no_parameters(self) -> None:
        """`head: cosine` puts the prototypes in the network, and then they ship with it."""
        assert list(build_loss(ComponentConfig(name="arcface"), facts("arcface")).parameters()) == []

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bfloat16", "float16"])
    def test_the_angle_is_taken_where_the_bound_that_keeps_it_finite_can_be_held(self, dtype: torch.dtype) -> None:
        """Half precision rounds the bound to exactly 1.0, and the derivative of arccos there is infinite.

        The loss reads a clean 0.0 while every gradient is NaN, so a run under mixed precision destroys
        its weights on the first step and reports nothing about it.
        """
        cosines = torch.eye(2, dtype=dtype, requires_grad=True)

        build_loss(ComponentConfig(name="arcface"), facts("arcface"))(cosines, torch.tensor([0, 1])).total.backward()

        assert cosines.grad is not None and bool(torch.isfinite(cosines.grad).all())

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bfloat16", "float16"])
    @pytest.mark.parametrize("name", sorted(ANGULAR))
    def test_the_angle_is_taken_in_single_precision_even_where_the_module_was_cast_to_half(
        self, name: str, dtype: torch.dtype
    ) -> None:
        """`precision=16-true` casts the parameters too, and an objective holding its own must still run.

        Upcasting only what arrives leaves a half prototype on the other side of the comparison, which
        is not a wrong number but a refusal to multiply at all — a run that dies on its first step.
        """
        objective = build_loss(ComponentConfig(name=name), facts(name)).to(dtype)
        outputs, targets = specimen(name)
        outputs = outputs.to(dtype).requires_grad_()

        objective(outputs, targets).total.backward()

        assert outputs.grad is not None and bool(torch.isfinite(outputs.grad).all())

    def test_a_target_a_mix_softened_is_refused_rather_than_flattened_into_zeros(self) -> None:
        """An angular margin is added to one identity, so it needs to be told which; a share is not one.

        Taken as it stands, `.long()` turns a distribution into zeros and broadcasting lets
        cross-entropy accept the result: two different mixes then report the very same number.
        """
        built = build_loss(ComponentConfig(name="arcface"), facts("arcface"))

        mixed = torch.tensor([[0.25, 0.75, 0.0], [0.75, 0.25, 0.0]])

        with pytest.raises(ValueError, match="identity"):
            built(torch.tensor([[0.9, -0.9, 0.0], [-0.9, 0.9, 0.0]]), mixed)

    def test_an_objective_reading_cosines_refuses_a_head_that_produces_something_else(self) -> None:
        """`head: linear` under this loss trains, saturates and reports a plausible number forever."""
        built = build_loss(ComponentConfig(name="arcface"), facts("arcface"))

        with pytest.raises(ValueError, match="cosines"):
            built(torch.randn(4, CLASSES) * 10, IDENTITIES)

    @pytest.mark.parametrize(
        ("declared", "refused"),
        [
            pytest.param({"margin": -0.1}, "radians", id="a margin that is not an angle"),
            pytest.param({"margin": 4.0}, "radians", id="a margin past half a turn"),
            pytest.param({"scale": 0.0}, "scale", id="a scale that cannot make a bounded score confident"),
        ],
    )
    def test_a_knob_that_could_not_do_its_work_is_refused_where_it_is_declared(
        self, declared: dict[str, object], refused: str
    ) -> None:
        with pytest.raises(ValueError, match=refused):
            build_loss(ComponentConfig.model_validate({"name": "arcface", **declared}), facts("arcface"))

    def test_an_objective_holding_prototypes_is_refused_on_a_target_with_no_identities(self) -> None:
        """Declared on a regression task it would otherwise build a zero-wide table and fail in a matmul."""
        with pytest.raises(ValueError, match="classes"):
            build_loss(ComponentConfig(name="arcface_proxy"), {**settled(), "embedding_dim": WIDTH})

    @pytest.mark.parametrize("restated", ["embedding_dim", "num_classes"])
    def test_a_width_the_task_settled_is_refused_where_a_declaration_restates_it(self, restated: str) -> None:
        with pytest.raises(ValueError, match=restated):
            build_loss(ComponentConfig.model_validate({"name": "arcface_proxy", restated: 9}), facts("arcface_proxy"))


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
        """The seam between a task and its loss: shapes, dtypes and the feature axis agree with no help."""
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

    def test_a_loss_that_needs_a_fact_the_task_has_not_got_is_named_with_what_it_was_offered(self) -> None:
        """The fact reaches the constructor either way; what is missing is said in the loss's own words."""
        with pytest.raises(ValueError, match="bins"):
            build_loss("expectation", {"semantics": None, "num_classes": None, "values": None})

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
