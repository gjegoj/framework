"""``record`` / ``facts``: what a target revealed goes in and comes out as one value."""

from __future__ import annotations

import pytest

from src.core import DataProfile, TargetFacts
from src.data import GaussianBinsTargetEncoder, LabelTargetEncoder, ScalarTargetEncoder


def test_a_record_survives_the_round_trip() -> None:
    profile = DataProfile()
    stored = TargetFacts(num_classes=3, class_names=["a", "b", "c"])

    profile.record("label", stored)

    assert profile.facts("label") == stored


def test_a_target_without_facts_reads_as_empty_and_refuses_a_class_count() -> None:
    """A profiled target that inferred nothing must not pass for one that has a count."""
    profile = DataProfile()

    profile.record("value", TargetFacts())

    assert profile.facts("value") == TargetFacts()
    with pytest.raises(LookupError, match="value"):
        profile.require_num_classes("value")


def test_records_of_different_tasks_do_not_mix() -> None:
    profile = DataProfile()

    profile.record("label", TargetFacts(num_classes=2))
    profile.record("score", TargetFacts(num_classes=5, class_values=[0.0, 1.0]))

    assert profile.facts("label").class_values is None
    assert profile.facts("score").num_classes == 5


def test_an_encoder_reports_exactly_what_it_inferred() -> None:
    """The caller never enumerates facts, so a new kind of fact reaches a profile untouched."""
    encoder = LabelTargetEncoder()
    encoder.fit(["cat", "dog"])

    assert encoder.facts() == TargetFacts(num_classes=2, class_names=["cat", "dog"])


def test_an_encoder_with_nothing_to_infer_reports_nothing() -> None:
    assert ScalarTargetEncoder().facts() == TargetFacts()


def test_a_binned_encoder_reports_the_values_behind_its_classes() -> None:
    encoder = GaussianBinsTargetEncoder(bins=4)
    encoder.fit([0.0, 1.0])

    facts = encoder.facts()

    assert facts.num_classes == 4
    assert facts.class_values is not None
    assert len(facts.class_values) == 4


def test_an_unfitted_encoder_reports_nothing_yet() -> None:
    """Facts appear at fit time; before that a profile would record a lie."""
    assert LabelTargetEncoder().facts() == TargetFacts()
