"""An input with no declared loader reads images — the common case of a vision framework."""

from __future__ import annotations

from typing import Any

from src.assembly.data import build_data_schema
from src.config.data import DEFAULT_INPUT_LOADER, InputColumnConfig
from src.data import ImageLoader
from src.data.registry import input_loader_registry
from tests.support.configs import DATA, paper_config


def schema_for(inputs: dict[str, Any]) -> Any:
    """The built schema of an experiment whose inputs are the declaration under test."""
    return build_data_schema(paper_config(data=DATA | {"inputs": inputs}))


def test_an_input_needs_nothing_but_its_column() -> None:
    schema = schema_for({"image": {"column": "image_path"}})

    assert isinstance(schema.inputs["image"].loader, ImageLoader)


def test_the_default_still_resolves_to_a_real_registered_loader() -> None:
    """The config layer names loaders, it does not import them; this keeps the name honest."""
    assert isinstance(input_loader_registry.create(DEFAULT_INPUT_LOADER), ImageLoader)


def test_a_declared_loader_keeps_its_arguments() -> None:
    """A root path is the ordinary reason to spell the loader out; the default must not eat it."""
    declared = InputColumnConfig.model_validate({"column": "image_path", "loader": {"name": "image", "root": "data"}})

    assert declared.loader.name == "image"
    assert declared.loader.params == {"root": "data"}


def test_a_non_image_input_declares_its_own_loader() -> None:
    """Multimodal runs stay expressible: the default is a convenience, not a restriction."""
    declared = InputColumnConfig.model_validate({"column": "text", "loader": {"_target_": "my_pkg.read_caption"}})

    assert declared.loader.target == "my_pkg.read_caption"
