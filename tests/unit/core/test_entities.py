"""The values every package exchanges: what they guarantee, and what they refuse."""

from __future__ import annotations

import math

import pytest
import torch

from src.core import (
    Bars,
    Batch,
    InputInfo,
    LossOutput,
    Matrix,
    Normalization,
    TargetInfo,
    TensorTree,
    class_name,
)
from src.core.entities import as_children, validate_classes, validate_name
from tests.unit.core.conftest import leaves


class TestNormalization:
    @pytest.mark.parametrize(
        ("mean", "std"),
        [
            pytest.param((), (), id="empty"),
            pytest.param((0.5, 0.5), (0.5,), id="lengths differ"),
            pytest.param((0.5,), (0.0,), id="zero std"),
            pytest.param((0.5,), (-1.0,), id="negative std"),
            pytest.param((math.nan,), (1.0,), id="nan mean"),
        ],
    )
    def test_needs_one_finite_mean_and_one_positive_std_per_channel(
        self, mean: tuple[float, ...], std: tuple[float, ...]
    ) -> None:
        with pytest.raises(ValueError):
            Normalization(mean, std)

    def test_an_input_declares_no_normalization_unless_it_says_so(self) -> None:
        assert InputInfo(shape=None).normalization is None
        assert InputInfo(shape=None, normalization=Normalization((0.5,), (0.5,))).normalization == Normalization(
            (0.5,), (0.5,)
        )


class TestBatch:
    def test_len_is_the_declared_count_not_a_tensor_length(self, batch: Batch) -> None:
        assert len(batch) == 2

    @pytest.mark.parametrize("count", [0, -1, True, 1.0], ids=["zero", "negative", "bool", "float"])
    def test_requires_a_positive_integer_count(self, count: object) -> None:
        with pytest.raises(ValueError, match="count"):
            Batch(inputs={}, count=count)  # type: ignore[arg-type]

    def test_moves_every_tensor_of_inputs_and_targets_and_keeps_the_rest(self, batch: Batch, tree: TensorTree) -> None:
        detached = batch.detach()

        assert len(leaves(detached.inputs["x"])) == len(leaves(tree))
        assert all(not leaf.requires_grad for leaf in leaves(detached.inputs["x"]))
        assert all(a.equal(b) for a, b in zip(leaves(detached.targets["y"]), leaves(batch.targets["y"]), strict=True))
        assert detached.count == batch.count
        assert detached.metadata == batch.metadata

    def test_to_returns_a_new_batch_on_the_requested_device(self, batch: Batch) -> None:
        moved = batch.to("cpu")

        assert moved is not batch
        assert all(leaf.device.type == "cpu" for leaf in leaves(moved.inputs["x"]))

    def test_is_immutable(self, batch: Batch) -> None:
        with pytest.raises(AttributeError):
            batch.count = 3  # type: ignore[misc]


class TestChildren:
    def test_a_mapping_becomes_the_children_of_a_module(self) -> None:
        children = as_children({"species": torch.nn.Identity()})

        assert isinstance(children, torch.nn.ModuleDict) and set(children) == {"species"}

    @pytest.mark.parametrize("reserved", ["training", "forward", "parameters"])
    def test_a_name_torch_keeps_for_itself_is_refused_as_the_name_it_came_from(self, reserved: str) -> None:
        """`validate_name` passes these: they are ordinary words, and the framework's own rules are met.

        Torch's are not — every module already answers to them — and a task carrying one reaches this
        point after the sources are read, as `KeyError: attribute 'training' already exists`, which
        names the collision without naming what a reader has to rename.
        """
        validate_name(reserved)

        with pytest.raises(ValueError, match=reserved):
            as_children({reserved: torch.nn.Identity()})


class TestClasses:
    @pytest.mark.parametrize(
        "classes",
        [
            pytest.param({}, id="empty"),
            pytest.param({1: "cat", 2: "dog"}, id="not from zero"),
            pytest.param({0: "cat", 2: "dog"}, id="gap"),
            pytest.param({"0": "cat"}, id="string index"),
            pytest.param({0: "cat", 1: " "}, id="blank name"),
            pytest.param({0: "cat", 1: 1}, id="non-string name"),
            pytest.param({0: "cat", 1: "cat"}, id="one name for two classes"),
            pytest.param({0: "cat", 1: "cat "}, id="one name once padded"),
        ],
    )
    def test_a_vocabulary_is_contiguous_from_zero_with_nonblank_names(self, classes: dict[object, object]) -> None:
        with pytest.raises(ValueError):
            validate_classes(classes)  # type: ignore[arg-type]

    def test_digits_are_a_word_like_any_other_here(self) -> None:
        """A vocabulary of bin centres names class 9 '10'; whether digits may be read as an index is
        the business of the one encoder that reads them that way."""
        validate_classes({index: f"{index + 1}" for index in range(10)})

    def test_target_info_counts_its_declared_classes(self) -> None:
        assert TargetInfo(classes={0: "cat", 1: "dog"}).num_classes == 2
        assert TargetInfo().num_classes is None

    def test_target_info_validates_the_vocabulary_it_is_given(self) -> None:
        with pytest.raises(ValueError):
            TargetInfo(classes={5: "cat"})

    def test_a_vocabulary_can_say_it_holds_only_what_the_training_split_showed(self) -> None:
        """What a run is judged on is then not what it learned, and the objective over it has no say there.

        Read by the composition root, which is the only place holding both the prepared data and the
        objectives built over it. Closed by default: every other vocabulary is declared whole.
        """
        assert TargetInfo(classes={0: "ann", 1: "bob"}, open_set=True).open_set
        assert not TargetInfo(classes={0: "cat", 1: "dog"}).open_set

    def test_an_open_vocabulary_with_no_vocabulary_at_all_is_unrepresentable(self) -> None:
        """Open means *these* were learned and others may arrive; with none learned there is no 'these'."""
        with pytest.raises(ValueError, match="open"):
            TargetInfo(open_set=True)


class TestLossOutput:
    @pytest.fixture
    def ce(self) -> LossOutput:
        """One term reporting itself, built as every leaf loss builds it: one value under both names."""
        value = torch.tensor(2.0)
        return LossOutput(value, losses={"ce": value}, contributions={"ce": value})

    @pytest.fixture
    def dice(self) -> LossOutput:
        value = torch.tensor(1.0)
        return LossOutput(value, losses={"dice": value}, contributions={"dice": value})

    def test_requires_a_scalar_total(self) -> None:
        with pytest.raises(ValueError, match="scalar"):
            LossOutput(torch.zeros(2))

    def test_addition_sums_totals_and_keeps_every_named_part(self, ce: LossOutput, dice: LossOutput) -> None:
        combined = ce + dice

        assert combined.total.item() == 3.0
        assert set(combined.losses) == set(combined.contributions) == {"ce", "dice"}

    def test_addition_refuses_colliding_names(self, ce: LossOutput) -> None:
        with pytest.raises(ValueError, match="ce"):
            _ = ce + ce

    def test_a_weight_scales_the_total_and_contributions_but_never_the_raw_losses(self, ce: LossOutput) -> None:
        weighted = 0.5 * ce

        assert weighted.total.item() == 1.0
        assert weighted.contributions["ce"].item() == 1.0
        assert weighted.losses["ce"].item() == 2.0
        assert (ce * 0.5).total.item() == weighted.total.item()

    @pytest.mark.parametrize("weight", [math.inf, math.nan])
    def test_refuses_a_non_finite_weight(self, ce: LossOutput, weight: float) -> None:
        with pytest.raises(ValueError):
            _ = ce * weight

    def test_a_weight_of_one_is_not_a_weighting(self, ce: LossOutput) -> None:
        """The default weight leaves the same values, so nothing downstream has two of them to tell apart."""
        assert 1.0 * ce is ce

    def test_a_report_shows_each_term_once_and_its_share_only_where_a_weight_moved_it(self, ce: LossOutput) -> None:
        """The share is a leaf of the term, so a namespaced name keeps the shape `<task>/<term>`."""
        assert set(ce.breakdown()) == {"ce"}
        assert set((0.5 * ce).breakdown()) == {"ce", "ce/contribution"}

    def test_a_share_is_the_weighted_value_and_the_term_stays_itself(self, ce: LossOutput) -> None:
        shown = (0.5 * ce).breakdown()

        assert shown["ce"].item() == 2.0
        assert shown["ce/contribution"].item() == 1.0

    def test_a_prefix_namespaces_the_parts_and_leaves_the_total(self, ce: LossOutput) -> None:
        scoped = ce.prefixed("label")

        assert set(scoped.losses) == set(scoped.contributions) == {"label/ce"}
        assert scoped.total is ce.total

    def test_prefixed_parts_from_two_tasks_add_without_collision(self, ce: LossOutput) -> None:
        assert set((ce.prefixed("a") + ce.prefixed("b")).losses) == {"a/ce", "b/ce"}


class TestMatrix:
    def test_a_reading_is_drawn_from_two_axes_and_says_so_when_it_is_not(self) -> None:
        with pytest.raises(ValueError, match="two axes"):
            Matrix(torch.zeros(3), xaxis="Predicted", yaxis="True")

    def test_labels_name_every_row_or_none_of_them(self) -> None:
        """Half a vocabulary on an axis draws a chart that reads plausibly and says the wrong thing."""
        with pytest.raises(ValueError, match="label"):
            Matrix(torch.eye(3), xaxis="Predicted", yaxis="True", labels=("cat", "dog"))

    def test_a_reading_may_be_drawn_without_naming_its_rows(self) -> None:
        assert Matrix(torch.eye(2), xaxis="Predicted", yaxis="True").labels is None


class TestNames:
    @pytest.mark.parametrize("name", ["label", "mask_path", "_", "p3", "species-2", "golden retriever", "ce@main"])
    def test_accepts_ordinary_identifiers(self, name: str) -> None:
        validate_name(name)

    @pytest.mark.parametrize("name", ["", " label", "label ", "a.b", "a/b"], ids=repr)
    def test_refuses_blank_padded_or_separator_bearing_names(self, name: str) -> None:
        with pytest.raises(ValueError):
            validate_name(name)

    def test_names_what_kind_of_name_it_was_given(self) -> None:
        with pytest.raises(ValueError, match="Task"):
            validate_name("a/b", label="Task")


class TestBars:
    def test_it_takes_a_row_of_values_per_series(self) -> None:
        with pytest.raises(ValueError, match="One row of bars per series"):
            Bars(series=("train", "val"), values=((1.0,),), labels=("cat",), xaxis="class", yaxis="count")

    def test_every_series_spans_the_same_labels(self) -> None:
        """A backend draws them as one grouped chart; a short row would silently shift every bar after it."""
        with pytest.raises(ValueError, match="same labels"):
            Bars(series=("train",), values=((1.0,),), labels=("cat", "dog"), xaxis="class", yaxis="count")


class TestClassName:
    def test_a_declared_class_is_called_what_the_vocabulary_calls_it(self) -> None:
        assert class_name({0: "cat", 1: "dog"}, 1) == "dog"

    @pytest.mark.parametrize(
        ("classes", "index"),
        [pytest.param(None, 2, id="a target with no vocabulary"), pytest.param({0: "cat"}, 3, id="past the end")],
    )
    def test_a_class_no_vocabulary_names_is_called_the_same_thing_by_every_reader(
        self, classes: dict[int, str] | None, index: int
    ) -> None:
        """A metric leaf in the tracker and the same class on a page: a run calling it `class3` in one
        place and `3` in the other would be describing two things."""
        assert class_name(classes, index) == f"class{index}"
