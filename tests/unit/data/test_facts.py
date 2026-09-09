"""What an encoder reports after fitting: the facts a head, a criterion and a page are sized from."""

from __future__ import annotations

from src.core import TaskFacts
from src.data import GaussianBinsTargetEncoder, LabelTargetEncoder, ScalarTargetEncoder


def test_an_encoder_reports_exactly_what_it_inferred() -> None:
    """The caller never enumerates facts, so a new kind of fact reaches a profile untouched."""
    encoder = LabelTargetEncoder(classes={0: "cat", 1: "dog"})
    encoder.fit(["cat", "dog"])

    assert encoder.facts() == TaskFacts(num_classes=2, class_names=("cat", "dog"))


def test_an_encoder_with_nothing_to_infer_reports_nothing() -> None:
    assert ScalarTargetEncoder().facts() == TaskFacts()


def test_a_binned_encoder_reports_the_values_behind_its_classes() -> None:
    """The bin count is incidental here, but it can no longer be an arbitrary small one.

    This read ``bins=4``, where a derived sigma leaves only 1.33 sigma of room and the
    lowest value's expectation lands 12.2% of the span inside it — measured. The encoder
    now refuses that rather than encoding it, so the number here is one that means what
    it says.
    """
    encoder = GaussianBinsTargetEncoder(bins=8)
    encoder.fit([0.0, 1.0])

    facts = encoder.facts()

    assert facts.num_classes == 8
    assert facts.class_values is not None
    assert len(facts.class_values) == 8
