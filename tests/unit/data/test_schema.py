"""``DataSchema`` contract: a declarative map from table columns to inputs and targets."""

from __future__ import annotations

import pytest
import torch

from src.data import DataSchema, InputColumn, ScalarTargetEncoder, TargetColumn


def test_schema_exposes_input_and_target_names() -> None:
    schema = DataSchema(
        inputs={"image": InputColumn(column="path", loader=torch.as_tensor)},
        targets={"score": TargetColumn(column="score", encoder=ScalarTargetEncoder())},
    )

    assert set(schema.inputs) == {"image"}
    assert set(schema.targets) == {"score"}


def test_schema_lists_every_referenced_column() -> None:
    schema = DataSchema(
        inputs={"image": InputColumn(column="path", loader=torch.as_tensor)},
        targets={"score": TargetColumn(column="score", encoder=ScalarTargetEncoder())},
    )

    assert schema.columns() == {"path", "score"}


def test_schema_requires_at_least_one_input() -> None:
    with pytest.raises(ValueError, match="input"):
        DataSchema(inputs={}, targets={})


def test_schema_allows_target_free_tasks() -> None:
    schema = DataSchema(
        inputs={"image": InputColumn(column="path", loader=torch.as_tensor)},
        targets={},
    )

    assert not schema.targets


def test_input_column_rejects_blank_column_name() -> None:
    with pytest.raises(ValueError, match="column"):
        InputColumn(column="  ", loader=torch.as_tensor)


def test_target_column_rejects_blank_column_name() -> None:
    with pytest.raises(ValueError, match="column"):
        TargetColumn(column="", encoder=ScalarTargetEncoder())
