"""A declaration becomes an object in one place; derived facts arrive from the caller, never from config."""

from __future__ import annotations

import pytest

from src.config import ComponentConfig
from src.config.instantiate import fill_signature, instantiate, instantiate_offering, resolve_factory
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

    def test_a_nested_name_is_refused_rather_than_handed_over_as_a_plain_mapping(
        self, registry: Registry[Widget]
    ) -> None:
        """The outer grammar teaches `name`, and one level down the same spelling means something else.

        A registry serves the position a section names and nothing below it, so a nested `name` reaches
        the constructor as the two-key mapping it literally is — which no constructor expects and none
        says anything about. Refused naming the spelling that does work there.
        """
        declared = {"name": "widget", "inner": {"name": "widget"}}

        with pytest.raises(ValueError, match="_target_"):
            instantiate(ComponentConfig.model_validate(declared), registry)


class TestWhatAConstructorWillTake:
    """A declaration meets the constructor it names: what is written has to be something that takes it."""

    @pytest.mark.parametrize("build", [instantiate, instantiate_offering], ids=["imposed", "offered"])
    def test_a_knob_the_constructor_does_not_take_is_refused_by_name(
        self, build: object, registry: Registry[Widget]
    ) -> None:
        """A knob one letter from a real one is refused the same way through either door.

        Both doors reach a constructor, and until now they answered differently: one let Python's own
        `TypeError` out with no declaration in it, the other read the same error as a fact the run had
        failed to settle and sent the reader off to declare one.
        """
        declared = ComponentConfig.model_validate({"name": "widget", "sizes": 3})

        with pytest.raises(ValueError, match="takes no sizes"):
            build(declared, registry)  # type: ignore[operator]

    def test_the_refusal_says_what_the_constructor_does_take(self, registry: Registry[Widget]) -> None:
        """Naming the alternatives is the whole of the fix: the reader is one word away and cannot see it."""
        with pytest.raises(ValueError, match="inner, size, tag"):
            instantiate(ComponentConfig.model_validate({"name": "widget", "sizes": 3}), registry)

    def test_a_fact_the_framework_imposes_is_refused_when_nothing_can_receive_it(
        self, registry: Registry[Widget]
    ) -> None:
        """An imposed fact is not the declaration's to write, so its absence is not the reader's to guess.

        A network of one's own reached by import path is handed its heads by the composition root; one
        whose constructor names no such parameter used to die in Python's words, which say what was
        passed and never that the framework is what passed it.
        """
        with pytest.raises(ValueError, match=r"hands .*model="):
            instantiate(ComponentConfig.model_validate("widget"), registry, model=object())

    def test_a_constructor_that_forwards_everything_is_taken_at_its_word(self) -> None:
        """Measured: torchmetrics, `smp.create_model` and an albumentations chain all forward what they are
        handed, and each refuses its own unknown knobs in its own words. Second-guessing them here would
        refuse declarations those libraries accept."""
        declared = {"_target_": "tests.unit.config.conftest.Anything", "whatever": 1}

        assert instantiate(ComponentConfig.model_validate(declared)).options == {"whatever": 1}

    def test_a_constructor_wanting_a_fact_no_run_settles_still_says_that(self) -> None:
        """The branch the misspelling used to be read as, left holding only what it is actually about."""
        declared = ComponentConfig.model_validate({"_target_": "tests.unit.config.conftest.Wanting"})

        with pytest.raises(ValueError, match="needs more"):
            instantiate_offering(declared, size=1)

    def test_a_derived_fact_is_refused_however_the_constructor_would_have_taken_it(self) -> None:
        """What the framework derives is not a declaration's to write, whether or not the constructor names it.

        A constructor forwarding everything takes the written copy and never sees the derived one, so the
        copy wins in silence — which is the whole of what this refusal exists to prevent. Filtering the
        facts by the signature before asking left exactly that case unasked.
        """
        declared = ComponentConfig.model_validate({"_target_": "tests.unit.config.conftest.Anything", "size": 3})

        with pytest.raises(ValueError, match="which the framework derives"):
            instantiate_offering(declared, size=7)

    def test_a_restated_fact_is_still_refused_as_restated_rather_than_as_unknown(
        self, registry: Registry[Widget]
    ) -> None:
        """Two refusals meet on one declaration, and the one naming the cause has to come first."""
        with pytest.raises(ValueError, match="which the framework derives"):
            instantiate(ComponentConfig.model_validate({"name": "widget", "size": 3}), registry, size=7)


class TestFillSignature:
    def test_hands_a_foreign_constructor_only_the_facts_it_names(self) -> None:
        def metric(num_classes: int, average: str = "macro") -> None: ...

        assert fill_signature(metric, num_classes=3, task="multiclass") == {"num_classes": 3}

    def test_never_reaches_a_constructor_that_forwards_everything(self) -> None:
        def anything(**kwargs: object) -> None: ...

        assert fill_signature(anything, num_classes=3) == {}
