"""A registry maps the names a config may write to the classes that serve them."""

from __future__ import annotations

import pytest

from src.core.registry import Registry


class Base:
    pass


@pytest.fixture
def registry() -> Registry[Base]:
    return Registry[Base]("loss")


def test_a_decorator_registers_and_returns_the_class_unchanged(registry: Registry[Base]) -> None:
    @registry.register("ce")
    class CrossEntropy(Base):
        pass

    assert registry.get("ce") is CrossEntropy
    assert CrossEntropy.__name__ == "CrossEntropy"


def test_the_names_it_lists_come_out_in_one_order(registry: Registry[Base]) -> None:
    """Tests parametrise over a registry, so the order it lists in is the order their ids appear in."""
    registry.register("focal")(Base)
    registry.register("cross_entropy")(Base)

    assert list(registry) == ["cross_entropy", "focal"]


def test_an_unknown_name_is_refused_with_the_known_names_listed(registry: Registry[Base]) -> None:
    registry.register("dice")(Base)

    with pytest.raises(LookupError, match=r"loss.*'focal'.*dice"):
        registry.get("focal")


def test_an_empty_registry_says_so(registry: Registry[Base]) -> None:
    with pytest.raises(LookupError, match="none"):
        registry.get("focal")


def test_a_name_is_registered_once(registry: Registry[Base]) -> None:
    registry.register("dice")(Base)

    with pytest.raises(ValueError, match="dice"):
        registry.register("dice")(Base)


def test_membership_reads_like_a_mapping(registry: Registry[Base]) -> None:
    registry.register("dice")(Base)

    assert "dice" in registry
    assert "focal" not in registry
