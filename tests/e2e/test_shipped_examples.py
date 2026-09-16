"""Every shipped example assembles into a run, over a table standing in for the one it names.

Composing a file proves its grammar and resolving its names proves they exist; neither builds anything.
Every refusal this framework makes while a run is assembled — a derived fact restated, two sections
disagreeing, a head that cannot read the stream it names, a chain that does not scale what it was
declared to — happens after both and is invisible to them. Without this an example can be shipped
broken under a green gate, which is how `contrastive` came to be.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from src.build import build
from src.config import ExperimentConfig, load_config
from tests.support.declarations import EXAMPLES, composed
from tests.support.table import write_table
from tests.support.text import text_family

ROWS = 16
"""Enough rows that no split is left empty by either rule the examples declare — measured: twelve is the
least that divides 0.7/0.15/0.15 three ways under both a stratified and a grouped split of this table."""


def local_weights(text: Path) -> Mapping[str, tuple[str, ...]]:
    """What each example needs so that assembling it fetches nothing from a model hub.

    Only the provenance of the weights is overridden, never what is built from them: every example
    below is assembled as its own file declares it. Where the network sits differs by the group the
    example overrides, so the paths are written out — a stale one ends the composition by name, which
    is what keeps this table from quietly going out of step with the groups.
    """
    hub = f"preprocessing.inputs.text.model_name={text}"
    timm_or_smp = ("model.backbone.pretrained=false",)
    return {
        "caption": (hub,),
        "classification": timm_or_smp,
        "contrastive": ("model.backbone.backbone.pretrained=false",),
        "finetuning": timm_or_smp,
        "metric_learning": timm_or_smp,
        "multitask": timm_or_smp,
        "pairing": ("model.backbone.encoders.image.pretrained=false", hub),
        "regression": timm_or_smp,
        "segmentation": timm_or_smp,
    }


REFUSED: Mapping[str, str] = {
    "contrastive": (
        "The scaling probe runs a stage's chain, and this example's evaluation stages draw views on "
        "purpose — the drawing is its supervision. Left as it is by the owner's decision of 2026-09-16."
    )
}
"""Examples known not to assemble, and why. Strict, so the day one of them does, this says so."""


@pytest.fixture(scope="module")
def table(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_table(tmp_path_factory.mktemp("rows"), samples=ROWS)


@pytest.fixture(scope="module")
def text(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return text_family(tmp_path_factory.mktemp("text") / "family")


def declared(example: str, *, table: Path, text: Path, directory: Path) -> ExperimentConfig:
    overrides = (f"data.source={table}", *local_weights(text)[example])
    return load_config(composed(f"experiment=examples/{example}", *overrides, directory=str(directory)))


@pytest.mark.parametrize(
    "example",
    [
        pytest.param(name, marks=pytest.mark.xfail(strict=True, reason=REFUSED[name])) if name in REFUSED else name
        for name in EXAMPLES
    ],
)
def test_a_shipped_example_assembles_the_run_it_declares(example: str, table: Path, text: Path, tmp_path: Path) -> None:
    built = build(declared(example, table=table, text=text, directory=tmp_path))

    assert built.module.learner.tasks, "a run that assembled is a run with something to learn"


def test_assembling_a_run_that_files_on_a_service_starts_nothing_there(
    table: Path, text: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One shipped example declares `tracker: clearml`, and the gate now assembles it on every run.

    The tracker creates the experiment at the first thing reported to it rather than in its constructor,
    which is held one step lower. What is held here is the step above: a logger handed to a `Trainer` is
    asked for its identity by some of Lightning's own paths, and asking starts the run. Were that to
    happen while assembling, every gate run would file an experiment on whatever service the machine is
    configured for. Shown by taking the client away — reaching for the service cannot then go unnoticed.
    """
    monkeypatch.setitem(sys.modules, "clearml", None)

    build(declared("classification", table=table, text=text, directory=tmp_path))


def test_every_example_in_the_repository_is_named_by_the_table_above(text: Path) -> None:
    """A table keyed by hand beside a globbed list: an example added and not named here would be skipped."""
    assert sorted(local_weights(text)) == EXAMPLES
