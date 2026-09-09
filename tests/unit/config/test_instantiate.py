"""The one grammar in action: registry names, import paths, recursion, loud failures."""

from __future__ import annotations

from typing import Any

import pytest

from src.config import ComponentConfig
from src.config.instantiate import instantiate, resolve_target
from src.core import Registry


class Greeter:
    """A stand-in component with one plain and one flag argument."""

    def __init__(self, name: str = "world", loud: bool = False) -> None:
        self.name = name
        self.loud = loud


class Pipeline:
    """A component whose parameter is itself a list of components."""

    def __init__(self, operations: list[object]) -> None:
        self.operations = operations


@pytest.fixture
def greeters() -> Registry[Greeter]:
    registry: Registry[Greeter] = Registry("greeter")
    registry.register("plain")(Greeter)
    return registry


def component(raw: object) -> ComponentConfig:
    return ComponentConfig.model_validate(raw)


def test_a_bare_name_builds_from_the_registry(greeters: Registry[Greeter]) -> None:
    assert isinstance(instantiate(component("plain"), greeters), Greeter)


def test_name_form_forwards_params(greeters: Registry[Greeter]) -> None:
    built = instantiate(component({"name": "plain", "loud": True}), greeters)

    assert built.loud is True


def test_target_form_imports_and_builds() -> None:
    built = instantiate(component({"_target_": "decimal.Decimal", "value": "1.5"}))

    assert str(built) == "1.5"


def test_dotted_paths_walk_submodules() -> None:
    resolved = resolve_target(component({"_target_": "torch.optim.lr_scheduler.StepLR"}))

    assert resolved.__name__ == "StepLR"


def test_resolve_target_does_not_build(greeters: Registry[Greeter]) -> None:
    """Some components must stay uninstantiated — an optimizer needs parameters later."""
    assert resolve_target(component("plain"), greeters) is Greeter


def test_nested_components_are_built_recursively() -> None:
    built = instantiate(
        component(
            {
                "_target_": f"{__name__}.Pipeline",
                "operations": [{"_target_": "decimal.Decimal", "value": "2"}],
            }
        )
    )

    assert [str(item) for item in built.operations] == ["2"]


def test_nested_components_work_under_the_registry_form_too(greeters: Registry[Greeter]) -> None:
    """Both forms take the same path, so recursion cannot differ between them."""
    registry: Registry[Pipeline] = Registry("pipeline")
    registry.register("pipeline")(Pipeline)

    built = instantiate(
        component({"name": "pipeline", "operations": [{"_target_": "decimal.Decimal", "value": "3"}]}),
        registry,
    )

    assert [str(item) for item in built.operations] == ["3"]


def test_a_mapping_of_components_builds_each_by_target() -> None:
    """``encoders: {image: {_target_: ...}, text: {_target_: ...}}`` — components keyed by name, as a dual encoder declares them."""
    built = instantiate(
        component(
            {
                "_target_": f"{__name__}.Pipeline",
                "operations": {"two": {"_target_": "decimal.Decimal", "value": "2"}, "plain": 5},
            }
        )
    )

    assert str(built.operations["two"]) == "2"
    assert built.operations["plain"] == 5


def test_a_nested_mapping_without_target_is_data_not_a_component() -> None:
    """Recursion triggers on ``_target_`` only: ``name`` means nothing without a registry."""
    registry: Registry[Pipeline] = Registry("pipeline")
    registry.register("pipeline")(Pipeline)

    built = instantiate(component({"name": "pipeline", "operations": [{"name": "just a value"}]}), registry)

    assert built.operations == [{"name": "just a value"}]


def test_unknown_registry_key_lists_the_registered_ones(greeters: Registry[Greeter]) -> None:
    with pytest.raises(LookupError, match="plain"):
        instantiate(component("typo"), greeters)


def test_a_name_without_a_registry_is_refused() -> None:
    with pytest.raises(LookupError, match="_target_"):
        instantiate(component("plain"))


def test_an_unimportable_target_names_the_path() -> None:
    # Hydra's own ImportError: `_target_` is Hydra's semantics, so is its diagnosis.
    with pytest.raises(ImportError, match="nope.Missing"):
        instantiate(component({"_target_": "nope.Missing"}))


def test_a_declaration_is_the_whole_of_what_reaches_the_constructor() -> None:
    """No value travels beside the declaration: a derived fact is passed by the caller, explicitly."""
    registry: Registry[Any] = Registry("takers")
    registry.register("takes")(lambda class_values=None: {"got": class_values})

    with pytest.raises(TypeError):
        instantiate(ComponentConfig(name="takes"), registry, class_values=[1.0])  # type: ignore[call-arg]
    assert instantiate(ComponentConfig(name="takes"), registry)["got"] is None
