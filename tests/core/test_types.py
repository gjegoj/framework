"""Shapes name their axes; a tree operation touches tensors and nothing else."""

from __future__ import annotations

import pytest
import torch

from src.core import Axis, TensorShape, TensorTree
from src.core.types import require_shape, require_tensor, tree_map
from tests.core.conftest import leaves


class TestTensorShape:
    def test_reads_a_size_by_axis_name(self) -> None:
        shape = TensorShape(axes=(Axis.CLASSES, Axis.HEIGHT, Axis.WIDTH), sizes=(3, None, None))

        assert shape.size(Axis.CLASSES) == 3
        assert shape.size(Axis.HEIGHT) is None

    def test_refuses_an_axis_it_does_not_have_by_name(self) -> None:
        shape = TensorShape(axes=(Axis.CHANNELS,), sizes=(512,))

        with pytest.raises(ValueError, match="tokens"):
            shape.size(Axis.TOKENS)

    def test_custom_axis_names_are_valid(self) -> None:
        assert TensorShape(axes=("mel", "time"), sizes=(80, None)).size("time") is None

    @pytest.mark.parametrize(
        ("axes", "sizes"),
        [
            pytest.param(("a", "b"), (1,), id="axes and sizes differ in length"),
            pytest.param(("a", "a"), (1, 1), id="duplicate axis"),
            pytest.param(("", "b"), (1, 1), id="blank axis"),
            pytest.param((" a",), (1,), id="padded axis"),
            pytest.param(("a",), (-1,), id="negative size"),
            pytest.param(("a",), (True,), id="boolean size"),
        ],
    )
    def test_refuses_an_inconsistent_declaration(self, axes: tuple[str, ...], sizes: tuple[object, ...]) -> None:
        with pytest.raises(ValueError):
            TensorShape(axes=axes, sizes=sizes)  # type: ignore[arg-type]


class TestTreeMap:
    def test_applies_the_operation_to_every_tensor_leaf(self, tree: TensorTree) -> None:
        moved = tree_map(lambda tensor: tensor + 1, tree)

        assert all(leaf.equal(original + 1) for leaf, original in zip(leaves(moved), leaves(tree), strict=True))

    def test_preserves_every_container_and_an_absent_leaf(self, tree: TensorTree) -> None:
        moved = tree_map(lambda tensor: tensor, tree)

        assert isinstance(moved, dict)
        assert isinstance(moved["text"], dict) and isinstance(moved["text"]["mask"], list)
        assert isinstance(moved["views"], tuple) and moved["views"][1] is None

    def test_a_bare_tensor_and_none_are_trees_too(self) -> None:
        assert require_tensor(tree_map(lambda tensor: tensor * 2, torch.ones(1)), name="doubled").item() == 2
        assert tree_map(lambda tensor: tensor, None) is None


def test_require_shape_narrows_one_shape_and_refuses_a_tree_by_name() -> None:
    shape = TensorShape(axes=("a",), sizes=(1,))

    assert require_shape(shape, name="pooled") is shape
    with pytest.raises(TypeError, match="pooled"):
        require_shape({"x": shape}, name="pooled")


class TestRequireTensor:
    def test_narrows_a_tensor(self) -> None:
        tensor = torch.zeros(1)

        assert require_tensor(tensor, name="logits") is tensor

    @pytest.mark.parametrize("value", [None, {"a": torch.zeros(1)}, [torch.zeros(1)]], ids=["none", "mapping", "list"])
    def test_refuses_a_structured_value_by_name(self, value: TensorTree) -> None:
        with pytest.raises(TypeError, match="logits"):
            require_tensor(value, name="logits")


def test_shape_is_hashable_and_comparable_by_value() -> None:
    one, other = (TensorShape(axes=("a",), sizes=(1,)) for _ in range(2))

    assert one == other and hash(one) == hash(other)
