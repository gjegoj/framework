"""An `export` declaration becomes the formats a run ships, with nothing written twice over."""

from __future__ import annotations

from typing import Any

import pytest

from src.config import ComponentConfig
from src.export.backends.onnx import OnnxExporter
from src.export.build import build_exporters

ONNX_CLASS = "src.export.backends.onnx.OnnxExporter"
TORCHSCRIPT_CLASS = "src.export.backends.torchscript.TorchScriptExporter"


def declaration(**declared: Any) -> ComponentConfig:
    return ComponentConfig.model_validate(declared)


def test_every_declared_format_is_built_in_the_order_it_was_declared() -> None:
    built = build_exporters([declaration(name="onnx"), declaration(name="pt2")])

    assert [one.suffix for one in built] == ["onnx", "pt2"]


def test_a_format_is_built_with_what_its_own_declaration_says() -> None:
    (built,) = build_exporters([declaration(name="onnx", opset=17)])

    assert isinstance(built, OnnxExporter)
    assert built.opset == 17


def test_two_declarations_that_would_write_one_file_are_refused_naming_both() -> None:
    """The legacy check compared declared names, so a registry name and an import path of the same class
    counted as two formats and wrote one file, the second silently replacing the first."""
    with pytest.raises(ValueError, match=r"\.onnx"):
        build_exporters([declaration(name="onnx"), declaration(_target_=ONNX_CLASS, opset=17)])


def test_a_declaration_that_builds_something_else_is_refused_by_name() -> None:
    with pytest.raises(TypeError, match="Path"):
        build_exporters([declaration(_target_="pathlib.Path")])


@pytest.mark.parametrize(
    ("format_name", "field", "wrong"),
    [("tensorrt", "onnx", TORCHSCRIPT_CLASS), ("ncnn", "torchscript", ONNX_CLASS)],
    ids=["an engine over something that is not an onnx graph", "an ncnn graph over something that is not traced"],
)
def test_a_format_written_through_another_refuses_a_graph_it_cannot_read(
    format_name: str, field: str, wrong: str
) -> None:
    """A nested position arrives through `params`, which nothing types, so no declaration owns this.

    The outer format reads a specific file — TensorRT parses ONNX, ncnn converts TorchScript — and a
    declaration naming the other one reaches the constructor untouched. Answered here rather than at the
    end of a run, on the one machine that has the library to answer it on.
    """
    declared = declaration(**{"name": format_name, field: {"_target_": wrong}})

    with pytest.raises(TypeError, match=field):
        build_exporters([declared])


@pytest.mark.parametrize(
    ("format_name", "field", "nested", "knob", "value"),
    [
        ("tensorrt", "onnx", ONNX_CLASS, "opset", 17),
        ("ncnn", "torchscript", TORCHSCRIPT_CLASS, "atol", 0.5),
    ],
    ids=["an engine over onnx", "an ncnn graph over torchscript"],
)
def test_a_format_written_through_another_keeps_what_its_declaration_said_about_that_one(
    format_name: str, field: str, nested: str, knob: str, value: object
) -> None:
    """Legacy constructed the intermediate exporter privately and dropped every option declared for it,
    so `{name: onnx, opset: 17}` beside `{name: tensorrt}` wrote one file at 17 and built the other at 18."""
    declared = declaration(**{"name": format_name, field: {"_target_": nested, knob: value}})

    (built,) = build_exporters([declared])

    assert getattr(getattr(built, field), knob) == value
