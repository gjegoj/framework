"""Binned target encoders: a continuous value as a distribution over ordered bins."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from src.data import GaussianBinsTargetEncoder, LinearBinsTargetEncoder
from src.data.registry import target_encoder_registry

TRAIN = [0.0, 25.0, 50.0, 75.0, 100.0]


def expectation(encoder: GaussianBinsTargetEncoder | LinearBinsTargetEncoder, value: float) -> float:
    values = encoder.class_values
    assert values is not None
    return float(np.asarray(encoder.encode(value)) @ np.asarray(values))


@pytest.mark.parametrize("encoder_type", [GaussianBinsTargetEncoder, LinearBinsTargetEncoder])
def test_the_encoding_is_a_distribution(encoder_type: type) -> None:
    encoder = encoder_type(bins=20)
    encoder.fit(TRAIN)

    encoded = encoder.encode(37.0)

    assert encoded.sum() == pytest.approx(1.0)
    assert (encoded >= 0).all()
    assert encoded.dtype == np.float32


@pytest.mark.parametrize("encoder_type", [GaussianBinsTargetEncoder, LinearBinsTargetEncoder])
@pytest.mark.parametrize("value", [0.0, 12.5, 50.0, 99.0, 100.0])
def test_the_encoded_distribution_still_means_the_value_it_came_from(encoder_type: type, value: float) -> None:
    """The property the padding buys: an expectation that survives the round trip.

    Bins ending at the data's extremes clip the mass of edge values, and the
    expectation of a clipped distribution drifts inward — silently regressing
    a model onto targets nobody declared.
    """
    encoder = encoder_type(bins=20)
    encoder.fit(TRAIN)

    assert expectation(encoder, value) == pytest.approx(value, abs=1.0)


@pytest.mark.parametrize("value", [0.0, 3.7, 62.4, 100.0])
def test_linear_bins_reproduce_the_value_exactly(value: float) -> None:
    """Two-point weights are exact inside the centre range; that is the trade they offer."""
    encoder = LinearBinsTargetEncoder(bins=20)
    encoder.fit(TRAIN)

    assert expectation(encoder, value) == pytest.approx(value, abs=1e-4)


def test_the_range_is_learned_from_the_training_split() -> None:
    encoder = GaussianBinsTargetEncoder(bins=10)
    encoder.fit([5.0, 15.0])

    values = encoder.class_values
    assert values is not None
    assert min(values) < 5.0 and max(values) > 15.0


def test_a_declared_range_is_used_as_given() -> None:
    encoder = GaussianBinsTargetEncoder(bins=10, low=0.0, high=1.0)
    encoder.fit([0.4, 0.6])

    assert expectation(encoder, 0.5) == pytest.approx(0.5, abs=0.05)


def test_the_bin_count_sizes_the_head_and_the_centres_travel_with_it() -> None:
    encoder = GaussianBinsTargetEncoder(bins=12)
    encoder.fit(TRAIN)

    assert encoder.num_classes == 12
    assert encoder.class_values is not None
    assert len(encoder.class_values) == 12


def test_sigma_defaults_to_one_bin_width_so_the_smoothing_actually_smooths() -> None:
    """Below roughly a third of the bin width the density collapses into one bin."""
    encoder = GaussianBinsTargetEncoder(bins=20)
    encoder.fit(TRAIN)

    assert encoder.encode(50.0).max() < 0.6


def test_a_sigma_far_below_the_bin_width_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    """It silently degrades to classification over quantised values, so it must be said."""
    with caplog.at_level(logging.WARNING):
        GaussianBinsTargetEncoder(bins=20, sigma=0.01).fit(TRAIN)

    assert "sigma" in caplog.text


def test_a_value_far_outside_the_range_lands_in_the_nearest_bin() -> None:
    """Validation rows are not bound by the training range; underflow must not yield NaN."""
    encoder = GaussianBinsTargetEncoder(bins=10, sigma=0.5, low=0.0, high=10.0)

    encoded = encoder.encode(1_000.0)

    assert encoded.sum() == pytest.approx(1.0)
    assert encoded.argmax() == len(encoded) - 1


def test_encoding_before_fitting_is_reported() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        GaussianBinsTargetEncoder().encode(1.0)


def test_a_constant_target_cannot_be_binned() -> None:
    with pytest.raises(ValueError, match="constant"):
        GaussianBinsTargetEncoder().fit([3.0, 3.0, 3.0])


def test_an_empty_training_split_is_reported() -> None:
    with pytest.raises(ValueError, match="empty"):
        GaussianBinsTargetEncoder().fit([])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bins": 1}, "at least 2 bins"),
        ({"low": 0.0}, "both"),
        ({"low": 1.0, "high": 0.0}, "low < high"),
        ({"sigma": -1.0}, "positive sigma"),
    ],
)
def test_contradictory_declarations_are_refused(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GaussianBinsTargetEncoder(**kwargs)  # type: ignore[arg-type]


def test_both_encoders_are_reachable_from_config() -> None:
    assert isinstance(target_encoder_registry.create("gaussian_bins"), GaussianBinsTargetEncoder)
    assert isinstance(target_encoder_registry.create("linear_bins"), LinearBinsTargetEncoder)
