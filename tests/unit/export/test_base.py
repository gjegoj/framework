"""What every deployment format answers before it writes anything: where its artifact goes."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
from torch import Tensor

from src.export import DeployableModel, Exporter, Runnable


class Writes(Exporter):
    """The contract with nothing behind it: these tests are about the rule, not about any one format."""

    suffix: ClassVar[str] = "onnx"

    def write(self, graph: DeployableModel, example: tuple[Tensor, ...], path: Path) -> None:
        raise NotImplementedError

    def load(self, path: Path) -> Runnable:
        raise NotImplementedError


def test_a_format_adds_its_suffix_rather_than_replacing_whatever_the_name_already_carries(tmp_path: Path) -> None:
    """A destination named after a version — `model.v2` — would lose the version to `with_suffix`."""
    assert Writes().artifact_path(tmp_path / "model.v2") == tmp_path / "model.v2.onnx"


@pytest.mark.parametrize(
    ("atol", "rtol"),
    [
        pytest.param(-1.0, 1e-3, id="a negative absolute allowance"),
        pytest.param(1e-4, -1.0, id="a negative relative allowance"),
        pytest.param(0.0, 0.0, id="no allowance at all"),
        pytest.param(float("nan"), 1e-3, id="an absolute allowance that is not a number"),
        pytest.param(1e-4, float("nan"), id="a relative allowance that is not a number"),
        pytest.param(float("inf"), 0.0, id="an allowance without a bound"),
    ],
)
def test_a_tolerance_that_could_prove_nothing_is_refused_where_it_is_declared(atol: float, rtol: float) -> None:
    """A negative allowance makes every share of it negative, so the worst value reads as zero used and a
    file nobody compared is reported as perfect; a zero one refuses an artifact that is exact."""
    with pytest.raises(ValueError, match="allowance"):
        Writes(atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "declared",
    [pytest.param({"atol": 1e-3}, id="an absolute allowance"), pytest.param({"rtol": 1e-2}, id="a relative one")],
)
def test_an_allowance_for_a_format_nothing_compares_is_refused(declared: dict[str, float]) -> None:
    """An allowance is what a comparison is judged by; beside `verify: false` it would be read by nothing."""
    with pytest.raises(ValueError, match="`verify: false`"):
        Writes(verify=False, **declared)


def test_asking_where_an_artifact_goes_writes_nothing(tmp_path: Path) -> None:
    """A question with a side effect leaves empty directories behind for every format a run considered."""
    destination = tmp_path / "artifacts" / "model"

    Writes().artifact_path(destination)

    assert not destination.parent.exists()
