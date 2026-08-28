"""A preset should carry its own target encoding; declaring it is an override, not a duty."""

from __future__ import annotations

from typing import Any

import pytest

from src.assembly.data import build_data_schema
from src.data import (
    BoxesTargetEncoder,
    LabelTargetEncoder,
    MaskTargetEncoder,
    MultiLabelTargetEncoder,
    ScalarTargetEncoder,
)
from tests.support.configs import paper_config


def schema_for(task: dict[str, Any]) -> Any:
    """The built schema of an experiment whose one task is the declaration under test."""
    return build_data_schema(paper_config(tasks={"target": task}))


@pytest.mark.parametrize(
    ("preset", "encoder"),
    [
        ("classification", LabelTargetEncoder),
        ("binary_classification", ScalarTargetEncoder),
        ("multilabel_classification", MultiLabelTargetEncoder),
        ("regression", ScalarTargetEncoder),
    ],
)
def test_a_preset_supplies_its_own_target_encoder(preset: str, encoder: type) -> None:
    vocabulary = {"classes": {0: "cat", 1: "dog"}} if preset in {"classification", "multilabel_classification"} else {}
    schema = schema_for({"preset": preset, "target": "y", **vocabulary})

    assert isinstance(schema.targets["target"].encoder, encoder)


def test_a_declared_encoder_still_wins() -> None:
    schema = schema_for(
        {
            "preset": "multilabel_classification",
            "target": "y",
            "classes": {0: "dog", 1: "cat"},
            "target_encoder": {"name": "multilabel", "separator": "|"},
        }
    )
    encoder = schema.targets["target"].encoder
    encoder.fit(["cat|dog"])

    assert encoder.class_names == ["dog", "cat"]


def test_a_segmentation_task_composes_the_mask_encoder_from_its_classes_alone() -> None:
    """The dense shape implies the encoder; the declared vocabulary carries the count."""
    schema = schema_for(
        {"preset": "segmentation", "target": "mask", "classes": {0: "pet", 1: "background", 2: "boundary"}}
    )
    encoder = schema.targets["target"].encoder

    assert isinstance(encoder, MaskTargetEncoder)
    assert encoder.num_classes == 3
    assert encoder.class_names == ["pet", "background", "boundary"]


@pytest.mark.parametrize("preset", ["classification", "multilabel_classification", "segmentation", "detection"])
def test_a_task_whose_encoder_reads_a_vocabulary_must_declare_classes(preset: str) -> None:
    """The index space is a declaration, never learned from whichever rows the split left in
    train — learned, it shrank when a rare class dropped out and the checkpoint stopped fitting."""
    with pytest.raises(ValueError, match="needs 'classes'"):
        schema_for({"preset": preset, "target": "y"})


def test_a_declared_encoder_without_classes_is_refused_the_same_way() -> None:
    with pytest.raises(ValueError, match="needs 'classes'"):
        schema_for({"preset": "classification", "target": "y", "target_encoder": {"name": "label"}})


def test_classes_written_inside_the_encoder_declaration_still_count() -> None:
    schema = schema_for(
        {
            "preset": "classification",
            "target": "y",
            "target_encoder": {"name": "label", "classes": {0: "cat", 1: "dog"}},
        }
    )

    assert schema.targets["target"].encoder.class_names == ["cat", "dog"]


def test_a_declared_mask_encoder_works_as_before() -> None:
    schema = schema_for(
        {
            "preset": "segmentation",
            "target": "mask",
            "classes": {0: "a", 1: "b", 2: "c"},
            "target_encoder": {"name": "mask"},
        }
    )

    assert isinstance(schema.targets["target"].encoder, MaskTargetEncoder)


def test_a_detection_task_needs_no_target_encoder_line() -> None:
    """INSTANCES admits no real encoder choice, and a knob with one correct value is code."""
    schema = schema_for({"preset": "detection", "target": "objects", "classes": {0: "cat", 1: "dog"}})

    assert isinstance(schema.targets["target"].encoder, BoxesTargetEncoder)


def test_a_structure_supervised_task_with_a_target_column_is_questioned() -> None:
    """Metric learning takes its supervision from the batch, so a target column needs explaining."""
    with pytest.raises(ValueError, match="target_encoder"):
        schema_for({"preset": "ranking", "target": "y"})


def test_a_structure_supervised_task_without_a_target_stays_targetless() -> None:
    schema = schema_for({"preset": "ranking"})

    assert schema.targets == {}


def test_declared_classes_reach_the_default_encoder() -> None:
    """The task's vocabulary travels to whichever encoder the objective implied."""
    schema = schema_for({"preset": "classification", "target": "y", "classes": {0: "cat", 1: "dog"}})

    assert schema.targets["target"].encoder.class_names == ["cat", "dog"]


def test_a_declaration_the_encoder_cannot_honour_is_refused() -> None:
    """Derived facts may be dropped silently; a user's declaration may not."""
    with pytest.raises(ValueError, match="vocabulary"):
        schema_for({"preset": "binary_classification", "target": "y", "classes": {0: "neg", 1: "pos"}})


def test_classes_declared_in_two_places_are_refused() -> None:
    """Two user-written vocabularies must not resolve silently; declare it once, on the task."""
    with pytest.raises(ValueError, match="once"):
        schema_for(
            {
                "preset": "classification",
                "target": "y",
                "classes": {0: "cat", 1: "dog"},
                "target_encoder": {"name": "label", "classes": {0: "dog", 1: "cat"}},
            }
        )
