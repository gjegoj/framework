"""``Registry`` contract: keyed factories with duplicate protection and named errors."""

from __future__ import annotations

import pytest

from src.core import Registry


class Greeter:
    def __init__(self, name: str = "world") -> None:
        self.name = name


def test_register_and_create_with_kwargs() -> None:
    registry: Registry[Greeter] = Registry("greeter")
    registry.register("plain")(Greeter)

    created = registry.create("plain", name="ds")

    assert isinstance(created, Greeter)
    assert created.name == "ds"


def test_register_works_as_a_decorator() -> None:
    registry: Registry[Greeter] = Registry("greeter")

    @registry.register("decorated")
    class Decorated(Greeter):
        pass

    assert isinstance(registry.create("decorated"), Decorated)


def test_duplicate_keys_are_rejected() -> None:
    registry: Registry[Greeter] = Registry("greeter")
    registry.register("plain")(Greeter)

    with pytest.raises(ValueError, match="already"):
        registry.register("plain")(Greeter)


def test_missing_key_error_lists_the_registered_ones() -> None:
    registry: Registry[Greeter] = Registry("greeter")
    registry.register("plain")(Greeter)

    with pytest.raises(LookupError, match="plain"):
        registry.create("typo")


def test_register_instance_returns_the_same_object() -> None:
    registry: Registry[Greeter] = Registry("greeter")
    shared = Greeter()
    registry.register_instance("shared", shared)

    assert registry.create("shared") is shared


def test_registry_iterates_over_its_keys() -> None:
    registry: Registry[Greeter] = Registry("greeter")
    registry.register("a")(Greeter)
    registry.register("b")(Greeter)

    assert set(registry) == {"a", "b"}
    assert "a" in registry


def test_hashable_composite_keys_work() -> None:
    registry: Registry[str] = Registry("pair")
    registry.register_instance(("global", "metric"), "arcface")

    assert registry.create(("global", "metric")) == "arcface"


def test_register_instance_as_a_decorator_serves_one_shared_object() -> None:
    """The decorator arity exists because Python decorates a class, never an expression."""
    registry: Registry[Greeter] = Registry("greeter")

    @registry.register_instance("shared")
    class Named(Greeter):
        pass

    assert registry.create("shared") is registry.create("shared")
    assert isinstance(registry.create("shared"), Named)


def test_a_declaration_that_cannot_build_dies_at_registration_not_first_use() -> None:
    """The decorator constructs eagerly, so a bad declaration fails at import —
    the same moment a passed instance would have — and registers nothing."""
    registry: Registry[Greeter] = Registry("greeter")

    with pytest.raises(ZeroDivisionError):

        @registry.register_instance("broken")
        class Broken(Greeter):
            def __init__(self) -> None:
                raise ZeroDivisionError

    assert "broken" not in registry
