"""Vocabulary pins for the taxonomy: axis members are a contract of the domain."""

from __future__ import annotations

from src.core import Geometry, InputTopology, Modality, Objective, OutputTopology, Stage, Stream


def test_stage_members_cover_the_training_lifecycle() -> None:
    assert [stage.value for stage in Stage] == ["train", "val", "test"]


def test_stage_compares_equal_to_its_string_value() -> None:
    assert Stage.TRAIN == "train"


def test_output_topology_members_cover_output_structures() -> None:
    assert {topology.value for topology in OutputTopology} == {
        "global",
        "dense",
        "instances",
    }


def test_input_topology_members_cover_input_arrangements() -> None:
    assert {topology.value for topology in InputTopology} == {
        "single",
        "multiview",
        "multistream",
    }


def test_objective_members_cover_label_semantics() -> None:
    assert {objective.value for objective in Objective} == {
        "multiclass",
        "binary",
        "multilabel",
        "continuous",
        "metric",
    }


def test_stream_names_the_standard_backbone_outputs() -> None:
    assert {stream.value for stream in Stream} == {
        "features",
        "encoder",
        "decoder",
        "logits",
        "embeddings",
    }


def test_modality_names_the_standard_inputs() -> None:
    assert {modality.value for modality in Modality} == {"image", "embedding", "text"}


def test_geometry_names_the_ways_a_value_rides_the_image() -> None:
    """The closed set the data layer declares and the transform seam consumes."""
    assert [member.value for member in Geometry] == ["none", "image", "mask", "boxes"]


def test_standard_names_stay_plain_strings() -> None:
    """Open vocabularies: enum members interoperate with custom string keys."""
    assert Stream.FEATURES == "features"
    assert Modality.IMAGE == "image"
