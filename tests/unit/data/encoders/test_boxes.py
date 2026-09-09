"""The boxes encoder: an objects cell into the pair the pipeline transforms, then ``Instances``."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
import torch

from src.core import Geometry, Instances
from src.data.encoders import BoxesTargetEncoder
from src.data.statistics import ClassDistribution

CELL: list[dict[str, Any]] = [
    {"box": [1.0, 2.0, 5.0, 6.0], "class": "dog"},
    {"box": [0.0, 0.0, 3.0, 3.0], "class": "cat"},
]


def fitted(cells: list[Any] | None = None) -> BoxesTargetEncoder:
    encoder = BoxesTargetEncoder(classes={0: "cat", 1: "dog"})
    encoder.fit(cells if cells is not None else [CELL])
    return encoder


def test_the_encoder_declares_that_its_values_are_boxes() -> None:
    """The marker ``build_stage_transforms`` reads to route the value through ``bbox_params``."""
    assert BoxesTargetEncoder.geometry is Geometry.BOXES


def test_the_declared_vocabulary_is_reported_as_facts_in_its_own_order() -> None:
    """A detection head sizes itself from the declaration; the order is the user's, not the alphabet's."""
    encoder = BoxesTargetEncoder(classes={0: "dog", 1: "cat"})

    assert encoder.class_names == ["dog", "cat"]
    assert encoder.facts().num_classes == 2


def test_a_declared_vocabulary_pins_the_index_space_and_refuses_an_invention() -> None:
    encoder = BoxesTargetEncoder(classes={0: "cat", 1: "dog"})

    with pytest.raises(LookupError, match="wolf"):
        encoder.fit([[{"box": [0.0, 0.0, 1.0, 1.0], "class": "wolf"}]])


def test_a_declared_vocabulary_survives_a_split_that_never_shows_a_class() -> None:
    """Learning from the rows would silently shrink the index space of a rare class."""
    encoder = BoxesTargetEncoder(classes={0: "cat", 1: "dog"})
    encoder.fit([[{"box": [0.0, 0.0, 1.0, 1.0], "class": "dog"}]])

    assert encoder.class_names == ["cat", "dog"]


def test_a_csv_cell_arrives_as_a_json_string_and_is_read_the_same() -> None:
    """One format, two carriers: a JSON table holds a list, a CSV cell holds its text."""
    from_text, names = fitted().load(json.dumps(CELL))
    from_list, listed_names = fitted().load(CELL)

    assert np.array_equal(from_text, from_list)
    assert names == listed_names == ["dog", "cat"]


def test_load_yields_the_pixel_pair_the_geometry_documents() -> None:
    boxes, names = fitted().load(CELL)

    assert boxes.shape == (2, 4)
    assert boxes.dtype == np.float32
    assert names == ["dog", "cat"]


def test_encode_turns_surviving_names_into_indices_in_a_per_sample_instances() -> None:
    """The encoder is this target's tensor boundary: a ragged value cannot wait for collation."""
    encoder = fitted()

    encoded = encoder.encode(encoder.load(CELL))

    assert isinstance(encoded, Instances)
    assert encoded.labels.tolist() == [1, 0]  # dog, cat against the sorted vocabulary
    assert torch.equal(encoded.sample_index, torch.zeros(2, dtype=torch.int64))
    assert encoded.boxes.dtype == torch.float32
    assert encoded.scores is None


def test_a_negative_cell_encodes_to_zero_rows() -> None:
    """An image with nothing in it is an observation, not a missing value."""
    encoder = fitted()

    encoded = encoder.encode(encoder.load([]))

    assert encoded.boxes.shape == (0, 4)
    assert encoded.labels.shape == (0,)


def test_an_unknown_class_at_encode_names_itself_and_the_known_ones() -> None:
    encoder = fitted()

    with pytest.raises(LookupError, match="wolf"):
        encoder.encode((np.zeros((1, 4), dtype=np.float32), ["wolf"]))


@pytest.mark.parametrize(
    "cell",
    [
        pytest.param([{"class": "dog"}], id="no box"),
        pytest.param([{"box": [1.0, 2.0, 3.0]}], id="no class, three corners"),
        pytest.param([{"box": [1.0, 2.0, 3.0], "class": "dog"}], id="three corners"),
    ],
)
def test_a_malformed_object_is_refused_showing_what_it_held(cell: Any) -> None:
    """Refused where it is read, not a thousand steps later inside a loss."""
    with pytest.raises(ValueError, match="box"):
        fitted().load(cell)


def test_a_cell_that_is_not_a_list_of_objects_at_all_is_refused_by_kind() -> None:
    """The wrong kind of value, not the wrong contents — the split ``require_tensor`` makes."""
    with pytest.raises(TypeError, match="list of objects"):
        fitted().load({"box": [1.0, 2.0, 3.0, 4.0]})


def test_the_distribution_counts_boxes_per_class_seeded_with_the_vocabulary() -> None:
    """Detection joins the standard dataset summary with no new reporting code."""
    encoder = BoxesTargetEncoder(classes={0: "cat", 1: "dog", 2: "fox"})
    encoder.fit([CELL])

    distribution = encoder.distribution([CELL, []])

    assert isinstance(distribution, ClassDistribution)
    assert distribution.counts == {"cat": 1, "dog": 1, "fox": 0}


@pytest.mark.parametrize(
    "box",
    [
        pytest.param([5.0, 2.0, 1.0, 6.0], id="x2 before x1"),
        pytest.param([1.0, 2.0, 1.0, 6.0], id="zero width"),
        pytest.param([-1.0, 2.0, 5.0, 6.0], id="negative corner"),
    ],
)
def test_a_box_with_no_area_or_off_the_origin_is_refused_at_fit_showing_the_cell(box: list[float]) -> None:
    """Refused at setup, where the cell can be shown. Measured on albumentationsx 2.3.7: left to
    the pipeline, the same box dies inside the first epoch as ``x_max is less than or equal to
    x_min for bbox [0.75 0.75 0.25 0.25 0.]`` — normalised, naming no row."""
    with pytest.raises(ValueError, match=r"x1 < x2.*dog"):
        BoxesTargetEncoder(classes={0: "cat", 1: "dog"}).fit([[{"box": box, "class": "dog"}]])


def test_a_box_that_is_not_four_numbers_is_refused_showing_what_it_held() -> None:
    """A string of corners iterates as characters; ``float(',')`` names nothing a user wrote."""
    with pytest.raises(ValueError, match="box.*1,2,5,6"):
        fitted().load([{"box": "1,2,5,6", "class": "dog"}])
