"""Every registered head projects the stream it reads onto the width it was asked for."""

from __future__ import annotations

import pytest
import torch

from src.models.registry import head_registry

READS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "linear": ((2, 8), (2, 3)),
    "conv": ((2, 8, 4, 4), (2, 3, 4, 4)),
    "cosine": ((2, 8), (2, 3)),
}
"""Name → the stream shape it reads at width 8, and the logits it returns for 3 classes."""


def test_every_registered_head_has_a_row_here() -> None:
    assert set(map(str, head_registry)) == set(READS)


@pytest.mark.parametrize("name", sorted(READS))
def test_it_projects_its_stream_onto_the_requested_width(name: str) -> None:
    given, expected = READS[name]

    logits = head_registry.create(name, in_features=8, out_features=3)(torch.randn(*given))

    assert tuple(logits.shape) == expected
