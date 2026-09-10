"""One grammar for every component: ``name`` or ``_target_``, every other key a constructor argument."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from src.config import ComponentConfig


@pytest.mark.parametrize(
    ("declared", "name", "target", "params"),
    [
        pytest.param("cross_entropy", "cross_entropy", None, {}, id="bare name"),
        pytest.param({"name": "dice", "smooth": 1.0}, "dice", None, {"smooth": 1.0}, id="name with arguments"),
        pytest.param({"_target_": "my.Loss", "gamma": 2}, None, "my.Loss", {"gamma": 2}, id="import path"),
        pytest.param(
            {"name": "x", "inner": {"_target_": "my.Inner"}},
            "x",
            None,
            {"inner": {"_target_": "my.Inner"}},
            id="nested stays raw",
        ),
    ],
)
def test_reads_every_spelling_the_same_way(
    declared: Any, name: str | None, target: str | None, params: dict[str, Any]
) -> None:
    component = ComponentConfig.model_validate(declared)

    assert (component.name, component.import_path, component.params) == (name, target, params)
    assert component.spelled == (name or target)


@pytest.mark.parametrize(
    "declared",
    [
        pytest.param({}, id="neither"),
        pytest.param({"name": "a", "_target_": "b"}, id="both"),
        pytest.param({"name": ""}, id="blank name"),
        pytest.param({"name": "a", "_partial_": True}, id="hydra meta key"),
        pytest.param({"name": "a", "_args_": [1]}, id="positional arguments"),
        pytest.param({"name": "a", "import_path": "b"}, id="alias spelled by field name"),
        pytest.param(42, id="not a mapping"),
    ],
)
def test_refuses_an_ambiguous_or_foreign_declaration(declared: Any) -> None:
    with pytest.raises(ValidationError):
        ComponentConfig.model_validate(declared)
