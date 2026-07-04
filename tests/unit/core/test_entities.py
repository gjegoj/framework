"""Unit tests for the framework-agnostic core (no torch/Lightning required)."""

import pytest

from src.core import Registry, RuntimeContext, Stage


class TestRegistry:
    def test_register_and_create(self) -> None:
        registry: Registry[str] = Registry("greeting")

        @registry.register("hello")
        def make(name: str) -> str:
            return f"hello {name}"

        assert "hello" in registry
        assert registry.create("hello", "world") == "hello world"
        assert list(registry.keys()) == ["hello"]

    def test_duplicate_key_raises(self) -> None:
        registry: Registry[int] = Registry("nums")
        registry.register("a")(lambda: 1)
        with pytest.raises(ValueError, match="already registered"):
            registry.register("a")(lambda: 2)

    def test_unknown_key_raises_with_available(self) -> None:
        registry: Registry[int] = Registry("nums")
        registry.register("a")(lambda: 1)
        with pytest.raises(KeyError, match="unknown key 'b'.*Available.*'a'"):
            registry.get("b")


class TestRuntimeContext:
    def test_num_classes_round_trips(self) -> None:
        ctx = RuntimeContext(num_classes={"species": 3})
        assert ctx.num_classes["species"] == 3

    def test_defaults_empty(self) -> None:
        assert RuntimeContext().num_classes == {}


def test_stage_is_str() -> None:
    assert Stage.TRAIN == "train"
    assert f"{Stage.VAL}" == "val"
