"""Unit tests for the generic brick-spec resolver (YAML mobility mechanism)."""

import pytest

from src.core.instantiate import instantiate, resolve_target
from src.core.registry import Registry


class _Widget:
    def __init__(self, size: int = 1, color: str = "red") -> None:
        self.size = size
        self.color = color


def _registry() -> Registry[_Widget]:
    reg: Registry[_Widget] = Registry("widget")
    reg.register("widget")(_Widget)
    return reg


class TestResolveTarget:
    def test_imports_dotted_path(self) -> None:
        cls = resolve_target("tests.test_instantiate._Widget")
        assert cls is _Widget

    def test_rejects_bare_name(self) -> None:
        with pytest.raises(ValueError, match="dotted path"):
            resolve_target("Widget")


class TestInstantiate:
    def test_string_spec_uses_registry_defaults(self) -> None:
        widget = instantiate("widget", _registry())
        assert widget.size == 1 and widget.color == "red"

    def test_mapping_name_spec_passes_params(self) -> None:
        widget = instantiate({"name": "widget", "size": 4}, _registry())
        assert widget.size == 4

    def test_injected_defaults_overridable_by_params(self) -> None:
        widget = instantiate({"name": "widget", "size": 9}, _registry(), size=2, color="blue")
        assert widget.size == 9 and widget.color == "blue"

    def test_target_escape_bypasses_registry(self) -> None:
        widget = instantiate(
            {"_target_": "tests.test_instantiate._Widget", "color": "green"},
            _registry(),
        )
        assert isinstance(widget, _Widget) and widget.color == "green"

    def test_mapping_without_name_or_target_raises(self) -> None:
        with pytest.raises(ValueError, match="needs a 'name' or '_target_'"):
            instantiate({"size": 3}, _registry())

    def test_non_spec_type_raises(self) -> None:
        with pytest.raises(TypeError):
            instantiate(42, _registry())  # type: ignore[arg-type]
