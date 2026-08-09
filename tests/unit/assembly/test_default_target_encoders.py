"""A preset should carry its own target encoding; declaring it is an override, not a duty."""

from __future__ import annotations

from typing import Any

import pytest

from src.assembly.data import build_data_schema
from src.data import (
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
    schema = schema_for({"preset": preset, "target": "y"})

    assert isinstance(schema.targets["target"].encoder, encoder)


def test_a_declared_encoder_still_wins() -> None:
    schema = schema_for(
        {
            "preset": "multilabel_classification",
            "target": "y",
            "target_encoder": {"name": "multilabel", "separator": "|"},
        }
    )
    encoder = schema.targets["target"].encoder
    encoder.fit(["cat|dog"])

    assert encoder.class_names == ["cat", "dog"]


def test_a_mask_target_cannot_be_guessed_and_says_so() -> None:
    """A dense target is an image of its own, and its encoder needs the class count."""
    with pytest.raises(ValueError, match="num_classes"):
        schema_for({"preset": "segmentation", "target": "mask"})


def test_a_declared_mask_encoder_works_as_before() -> None:
    schema = schema_for(
        {"preset": "segmentation", "target": "mask", "target_encoder": {"name": "mask", "num_classes": 3}}
    )

    assert isinstance(schema.targets["target"].encoder, MaskTargetEncoder)


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
