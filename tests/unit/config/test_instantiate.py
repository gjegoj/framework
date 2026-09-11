"""A declaration becomes an object in one place; derived facts arrive from the caller, never from config."""

from __future__ import annotations

import pytest

from src.config import ComponentConfig
from src.config.instantiate import fill_signature, instantiate, resolve_factory
from src.core import Registry
from tests.unit.config.conftest import Widget


class TestResolveTarget:
    def test_a_name_resolves_through_the_registry(self, registry: Registry[Widget]) -> None:
        assert resolve_factory(ComponentConfig.model_validate("widget"), registry) is Widget

    def test_an_import_path_resolves_without_a_registry(self) -> None:
        assert (
            resolve_factory(ComponentConfig.model_validate({"_target_": "tests.unit.config.conftest.Widget"})) is Widget
        )

    def test_a_name_without_a_registry_is_refused_naming_the_alternative(self) -> None:
        with pytest.raises(LookupError, match="_target_"):
            resolve_factory(ComponentConfig.model_validate("widget"))

    @pytest.mark.parametrize(
        "target", ["tests.unit.config.conftest.Missing", "no_such_module.Thing", "tests.unit.config.conftest"]
    )
    def test_a_path_that_is_not_a_constructor_is_refused_by_name(self, target: str) -> None:
        with pytest.raises((LookupError, TypeError), match=target.rsplit(".", 1)[-1]):
            resolve_factory(ComponentConfig.model_validate({"_target_": target}))


class TestInstantiate:
    def test_builds_with_the_declared_arguments(self, registry: Registry[Widget]) -> None:
        widget = instantiate(ComponentConfig.model_validate({"name": "widget", "size": 3}), registry)

        assert (widget.size, widget.tag) == (3, "")

    def test_facts_from_the_caller_reach_the_constructor(self, registry: Registry[Widget]) -> None:
        widget = instantiate(ComponentConfig.model_validate("widget"), registry, size=7)

        assert widget.size == 7

    def test_a_fact_restated_in_config_is_refused_by_name(self, registry: Registry[Widget]) -> None:
        with pytest.raises(ValueError, match=r"widget.*size"):
            instantiate(ComponentConfig.model_validate({"name": "widget", "size": 3}), registry, size=7)

    def test_a_nested_import_path_is_built_first(self, registry: Registry[Widget]) -> None:
        declared = {"name": "widget", "inner": {"_target_": "tests.unit.config.conftest.Widget", "tag": "inner"}}

        widget = instantiate(ComponentConfig.model_validate(declared), registry)

        assert isinstance(widget.inner, Widget) and widget.inner.tag == "inner"

    def test_nested_values_without_a_target_stay_plain(self, registry: Registry[Widget]) -> None:
        declared = {"name": "widget", "inner": {"a": [1, {"b": 2}]}}

        assert instantiate(ComponentConfig.model_validate(declared), registry).inner == {"a": [1, {"b": 2}]}

    def test_a_nested_name_is_a_plain_value_because_a_nested_position_has_no_registry(
        self, registry: Registry[Widget]
    ) -> None:
        widget = instantiate(ComponentConfig.model_validate({"name": "widget", "inner": {"name": "widget"}}), registry)

        assert widget.inner == {"name": "widget"}


class TestFillSignature:
    def test_hands_a_foreign_constructor_only_the_facts_it_names(self) -> None:
        def metric(num_classes: int, average: str = "macro") -> None: ...

        assert fill_signature(metric, num_classes=3, task="multiclass") == {"num_classes": 3}

    def test_never_reaches_a_constructor_that_forwards_everything(self) -> None:
        def anything(**kwargs: object) -> None: ...

        assert fill_signature(anything, num_classes=3) == {}
