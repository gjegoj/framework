"""Holding part of a model still while the rest learns, and letting it go when the run says."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from src.callbacks.freeze import Freeze
from tests.unit.callbacks.conftest import prepared, statistics_of, weights_of


def freezing(**declared: Any) -> list[dict[str, Any]]:
    return [{"name": "freeze", "modules": ["backbone"], **declared}]


def test_what_is_named_is_held_still_while_the_rest_learns(declaration: Mapping[str, Any]) -> None:
    """Declared with ``train_bn: false``, so that held still means all of it: normalisation keeps
    learning by default, and what it learns has a test of its own below."""
    built = prepared(declaration, callbacks=freezing(train_bn=False))
    model = built.module.learner.model
    backbone, head = weights_of(model.get_submodule("backbone")), weights_of(model.get_submodule("heads.species"))

    built.trainer.fit(built.module, datamodule=built.data)

    assert torch.equal(weights_of(model.get_submodule("backbone")), backbone), "the frozen part learned anyway"
    assert not torch.equal(weights_of(model.get_submodule("heads.species")), head), "and the rest learned nothing"


def test_it_lets_go_at_the_epoch_the_run_declared(declaration: Mapping[str, Any]) -> None:
    """Half of a two-epoch run is the first whole epoch at that share, so the second one trains it.

    Held with ``train_bn: false`` so that the only thing able to move these weights is being let go.
    """
    built = prepared(declaration, epochs=2, callbacks=freezing(until=0.5, train_bn=False))
    backbone = weights_of(built.module.learner.model.get_submodule("backbone"))

    built.trainer.fit(built.module, datamodule=built.data)

    assert not torch.equal(weights_of(built.module.learner.model.get_submodule("backbone")), backbone)


@pytest.mark.parametrize(
    ("train_bn", "moves"),
    [pytest.param(True, True, id="kept learning"), pytest.param(False, False, id="held with the rest")],
)
def test_normalisation_may_keep_learning_this_datasets_statistics(
    declaration: Mapping[str, Any], train_bn: bool, moves: bool
) -> None:
    """Those statistics describe *this* data, not the one the frozen weights were trained on."""
    built = prepared(declaration, callbacks=freezing(train_bn=train_bn))
    before = statistics_of(built.module.learner.model.get_submodule("backbone"))

    built.trainer.fit(built.module, datamodule=built.data)

    assert torch.equal(statistics_of(built.module.learner.model.get_submodule("backbone")), before) is not moves


def test_a_path_that_names_nothing_says_what_is_there_instead(declaration: Mapping[str, Any]) -> None:
    built = prepared(declaration, callbacks=freezing(modules=["trunk"]))

    with pytest.raises(LookupError, match="backbone"):
        built.trainer.fit(built.module, datamodule=built.data)


def test_a_path_reaching_something_that_is_not_a_module_is_refused(declaration: Mapping[str, Any]) -> None:
    built = prepared(declaration, callbacks=freezing(modules=["backbone.feature_shapes"]))

    with pytest.raises(LookupError, match="feature_shapes"):
        built.trainer.fit(built.module, datamodule=built.data)


def test_freezing_nothing_is_refused_where_it_was_declared() -> None:
    with pytest.raises(ValueError, match="named"):
        Freeze(modules=[])


def held(module: nn.Module) -> set[bool]:
    """Whether each parameter is still held, as a set — one value means the whole part agrees."""
    return {parameter.requires_grad for parameter in module.parameters()}


def test_a_run_continued_past_the_release_lets_go_as_an_uninterrupted_one_did(
    declaration: Mapping[str, Any], tmp_path: Path
) -> None:
    """``BaseFinetuning`` freezes again from every ``setup``, and a continued run starts after the
    epoch the release was tied to — so it would be held for the rest of its life, and say nothing."""
    saving = {"name": "checkpoint", "save_top_k": -1, "dirpath": str(tmp_path / "kept")}
    declared = [*freezing(until=0.5, train_bn=False), saving]
    first = prepared(declaration, epochs=4, callbacks=declared)
    first.trainer.fit(first.module, datamodule=first.data)
    # One epoch short of the end, which is also past the release: the case a crash actually leaves.
    written = sorted((tmp_path / "kept").glob("*.ckpt"))[-2]

    resumed = prepared(declaration, epochs=4, callbacks=declared)
    resumed.trainer.fit(resumed.module, datamodule=resumed.data, ckpt_path=str(written))

    assert held(resumed.module.learner.model.get_submodule("backbone")) == {True}
