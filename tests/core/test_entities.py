"""The values every package exchanges: what they guarantee, and what they refuse."""

from __future__ import annotations

import math

import pytest
import torch

from src.core import Batch, InputInfo, LossOutput, Normalization, TargetInfo, TensorTree
from src.core.entities import validate_classes, validate_name
from tests.core.conftest import leaves


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
        ],
    )
    def test_a_vocabulary_is_contiguous_from_zero_with_nonblank_names(self, classes: dict[object, object]) -> None:
        with pytest.raises(ValueError):
            validate_classes(classes)  # type: ignore[arg-type]

    def test_target_info_counts_its_declared_classes(self) -> None:
        assert TargetInfo(classes={0: "cat", 1: "dog"}).num_classes == 2
        assert TargetInfo().num_classes is None

    def test_target_info_validates_the_vocabulary_it_is_given(self) -> None:
        with pytest.raises(ValueError):
            TargetInfo(classes={5: "cat"})


class TestLossOutput:
    @pytest.fixture
    def ce(self) -> LossOutput:
        return LossOutput(torch.tensor(2.0), losses={"ce": torch.tensor(2.0)}, contributions={"ce": torch.tensor(2.0)})

    @pytest.fixture
    def dice(self) -> LossOutput:
        return LossOutput(
            torch.tensor(1.0), losses={"dice": torch.tensor(1.0)}, contributions={"dice": torch.tensor(1.0)}
        )

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

    def test_a_prefix_namespaces_the_parts_and_leaves_the_total(self, ce: LossOutput) -> None:
        scoped = ce.prefixed("label")

        assert set(scoped.losses) == set(scoped.contributions) == {"label/ce"}
        assert scoped.total is ce.total

    def test_prefixed_parts_from_two_tasks_add_without_collision(self, ce: LossOutput) -> None:
        assert set((ce.prefixed("a") + ce.prefixed("b")).losses) == {"a/ce", "b/ce"}


class TestNames:
    @pytest.mark.parametrize("name", ["label", "mask_path", "_", "p3", "species-2", "golden retriever"])
    def test_accepts_ordinary_identifiers(self, name: str) -> None:
        validate_name(name)

    @pytest.mark.parametrize("name", ["", " label", "label ", "a.b", "a/b", "a@b"], ids=repr)
    def test_refuses_blank_padded_or_separator_bearing_names(self, name: str) -> None:
        with pytest.raises(ValueError):
            validate_name(name)

    def test_names_the_kind_in_its_message(self) -> None:
        with pytest.raises(ValueError, match="Task"):
            validate_name("a/b", kind="Task")
