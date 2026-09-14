"""Weights that came from somewhere else: what fits is filled, what does not is named rather than dropped."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from torch import nn

from src.models.weights import start_from

if TYPE_CHECKING:
    from pathlib import Path

WIDTH, CLASSES = 4, 3


class Wrapped(nn.Module):
    """A network holding a library's own graph one attribute in, which is what ``TimmBackbone`` does."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Linear(WIDTH, CLASSES)


def architecture(path: Path, **extra: torch.Tensor) -> str:
    """A file naming the library's graph, as a pretraining script or a model hub writes one."""
    torch.save({"weight": torch.ones(CLASSES, WIDTH), "bias": torch.ones(CLASSES), **extra}, path)
    return str(path)


def test_a_file_that_names_the_architecture_fills_the_network_that_holds_it(tmp_path: Path) -> None:
    """A file names the library's own graph; this framework holds that graph one attribute in.

    Measured on timm 1.0: a resnet18 file and this framework's backbone for it share none of their 120
    names, and all 120 once that one attribute is accounted for. A declaration naming a file cannot be
    asked to know where the family put its network.
    """
    network = Wrapped()

    start_from(network, architecture(tmp_path / "b.pth"), inside="model.")

    assert torch.equal(network.model.weight, torch.ones(CLASSES, WIDTH))


def test_what_the_file_carried_that_this_network_has_no_place_for_is_handed_back(tmp_path: Path) -> None:
    """A trained file carries a classifier, and a backbone built headless has nowhere to put it.

    Handed back rather than dropped, because those rows are what a run growing its class space starts
    from; handed back rather than loaded, because this network does not have them.
    """
    network = Wrapped()
    path = architecture(tmp_path / "b.pth", **{"fc.weight": torch.zeros(7, CLASSES), "fc.bias": torch.zeros(7)})

    carried = start_from(network, path, inside="model.", aside=("fc.",))

    assert sorted(carried) == ["fc.bias", "fc.weight"]


def test_a_file_that_shares_no_name_with_this_network_is_refused_by_the_declaration(tmp_path: Path) -> None:
    """A file of another architecture loads nothing at all, and `strict: false` would say so with silence.

    Measured on timm 1.0: loading a foreign file with torch's own leniency moves 0 of 120 tensors and
    reports success, after which the run trains a random encoder and every number looks ordinary.
    """
    torch.save({"blocks.0.attn.qkv.weight": torch.ones(2, 2)}, tmp_path / "vit.pth")

    with pytest.raises(ValueError, match="shares no"):
        start_from(Wrapped(), str(tmp_path / "vit.pth"), inside="model.")


@pytest.mark.parametrize(
    ("held", "named"),
    [
        pytest.param(
            {"weight": torch.ones(CLASSES, WIDTH)},
            r"only in part — it carries nothing for model\.bias\.",
            id="a file that fills only half of it",
        ),
        pytest.param(
            {"weight": torch.ones(CLASSES, WIDTH), "bias": torch.ones(CLASSES), "extra": torch.ones(CLASSES)},
            r"only in part — this network has no place for model\.extra\.",
            id="a file bringing a name this network has nowhere to put",
        ),
        pytest.param(
            {"weight": torch.ones(CLASSES, WIDTH), "extra": torch.ones(CLASSES)},
            r"carries nothing for model\.bias, and this network has no place for model\.extra\.",
            id="a file that is wrong in both directions at once",
        ),
    ],
)
def test_a_file_that_fits_only_partly_is_refused_rather_than_half_loaded(
    held: dict[str, torch.Tensor], named: str, tmp_path: Path
) -> None:
    """The rule the rest of this framework already keeps: a network half from a file is not a network.

    Each case is matched on the whole clause the refusal owes it, rather than on a name the file
    happens to hold: measured, matching `bias` alone was satisfied by the refusal *above* — which
    prints the names this network is built of — so the test passed while the wrong thing was said.
    A file that fits everything and brings extras is the smp case, where a family carries a head the
    declaration did not name as one to hold back.
    """
    torch.save(held, tmp_path / "part.pth")

    with pytest.raises(ValueError, match=named):
        start_from(Wrapped(), str(tmp_path / "part.pth"), inside="model.")


def test_a_file_that_is_not_weights_at_all_is_refused_before_anything_is_put_anywhere(tmp_path: Path) -> None:
    """A path is written by hand, and what it points at is as likely to be a table or a log as a network."""
    torch.save({"epoch": 3}, tmp_path / "notes.pth")

    with pytest.raises(ValueError, match="holds no weights"):
        start_from(Wrapped(), str(tmp_path / "notes.pth"), inside="model.")
