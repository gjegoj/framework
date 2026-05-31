"""Unit tests for the framework-agnostic core (no torch/Lightning required)."""

import pytest

from src.core import Registry, RuntimeContext, RuntimeValue, Stage, resolve_runtime


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
    def test_resolve_per_task_num_classes(self) -> None:
        ctx = RuntimeContext(num_classes={"species": 3})
        assert ctx.resolve(RuntimeValue("num_classes", task="species")) == 3

    def test_resolve_scalar_field(self) -> None:
        ctx = RuntimeContext(total_steps=1000)
        assert ctx.resolve(RuntimeValue("total_steps")) == 1000

    def test_num_classes_requires_task(self) -> None:
        ctx = RuntimeContext(num_classes={"species": 3})
        with pytest.raises(ValueError, match="requires a task name"):
            ctx.resolve(RuntimeValue("num_classes"))

    def test_unpopulated_value_raises(self) -> None:
        ctx = RuntimeContext()
        with pytest.raises(ValueError, match="not populated yet"):
            ctx.resolve(RuntimeValue("total_steps"))

    def test_unknown_task_raises(self) -> None:
        ctx = RuntimeContext(num_classes={"species": 3})
        with pytest.raises(KeyError, match="not inferred for task 'color'"):
            ctx.resolve(RuntimeValue("num_classes", task="color"))


class TestResolveRuntime:
    def test_nested_structure_resolved(self) -> None:
        ctx = RuntimeContext(num_classes={"a": 5}, total_steps=200)
        spec = {
            "out_features": RuntimeValue("num_classes", task="a"),
            "schedule": [RuntimeValue("total_steps"), 0.1],
            "static": "keep",
        }
        resolved = resolve_runtime(spec, ctx)
        assert resolved == {"out_features": 5, "schedule": [200, 0.1], "static": "keep"}

    def test_plain_value_untouched(self) -> None:
        ctx = RuntimeContext()
        assert resolve_runtime(42, ctx) == 42


def test_stage_is_str() -> None:
    assert Stage.TRAIN == "train"
    assert f"{Stage.VAL}" == "val"
