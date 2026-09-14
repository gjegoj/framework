"""One grammar for every component: ``name`` or ``_target_``, every other key a constructor argument."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from src.config import ComponentConfig, HeadConfig


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


class TestHead:
    """A head declares the features it reads: one, or several where its task is learned over a pair of them."""

    @pytest.mark.parametrize(
        ("declared", "streams"),
        [
            pytest.param({"name": "linear", "stream": "pooled"}, ("pooled",), id="one"),
            pytest.param(
                {"name": "linear", "stream": ["image_pooled", "text_pooled"]},
                ("image_pooled", "text_pooled"),
                id="several, in the order they were written",
            ),
            pytest.param({"name": "native"}, (), id="none, for the task's own default to fill in"),
        ],
    )
    def test_reads_one_name_or_several_the_same_way(self, declared: Any, streams: tuple[str, ...]) -> None:
        """Whichever shape the declaration took, what builds the head reads one: `stream: pooled` is a list of one."""
        assert HeadConfig.model_validate(declared).streams == streams

    @pytest.mark.parametrize(
        "stream",
        [
            pytest.param(" pooled", id="padded"),
            pytest.param(["image_pooled", " text_pooled"], id="padded among several"),
            pytest.param(["pooled", "pooled"], id="one feature named twice"),
            pytest.param([], id="a list naming nothing"),
        ],
    )
    def test_refuses_anything_but_distinct_names_of_features(self, stream: Any) -> None:
        """A feature named twice would build two heads over one stream, which can only be a slip of the pen."""
        with pytest.raises(ValidationError):
            HeadConfig(name="linear", stream=stream)
