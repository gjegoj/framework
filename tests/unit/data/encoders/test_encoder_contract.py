"""Every registered target encoder, against the port: fit returns it, load then encode yields a value, the facts agree."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from src.core import Geometry
from src.data.registry import target_encoder_registry

CLASSES = {0: "cat", 1: "dog"}

DECLARED: dict[str, tuple[dict[str, Any], list[Any]]] = {
    "label": ({"classes": CLASSES}, ["cat", "dog"]),
    "multilabel": ({"classes": CLASSES}, ["cat,dog", "dog"]),
    "boxes": ({"classes": CLASSES}, [[{"box": [0.0, 0.0, 2.0, 2.0], "class": "cat"}], []]),
    "mask": ({"classes": CLASSES}, ["mask.png"]),
    "scalar": ({}, [0.0, 1.0, 2.0]),
    "gaussian_bins": ({"bins": 8}, [0.0, 1.0, 2.0]),
    "linear_bins": ({"bins": 4}, [0.0, 1.0, 2.0]),
}
"""Name → the smallest arguments it constructs from, and cells a table would hold for it."""


def declared(name: str, root: Path) -> tuple[Any, list[Any]]:
    arguments, cells = DECLARED[name]
    if name == "mask":
        cv2.imwrite(str(root / "mask.png"), np.array([[0, 1], [1, 0]], dtype=np.uint8))
        arguments = arguments | {"root": root}
    return target_encoder_registry.create(name, **arguments), cells


def test_every_registered_encoder_has_a_row_here() -> None:
    """A new encoder joins the contract by declaring how it constructs — or this fails naming it."""
    assert set(map(str, target_encoder_registry)) == set(DECLARED)


@pytest.mark.parametrize("name", sorted(DECLARED))
def test_fit_returns_the_encoder_and_load_then_encode_yields_a_value(name: str, tmp_path: Path) -> None:
    encoder, cells = declared(name, tmp_path)

    fitted = encoder.fit(cells)
    encoder.validate(cells)
    value = encoder.encode(encoder.load(cells[0]))

    assert fitted is encoder
    assert value is not None


@pytest.mark.parametrize("name", sorted(DECLARED))
def test_the_class_facts_agree_with_each_other(name: str, tmp_path: Path) -> None:
    """A vocabulary or a bin layout implies a class count of the same length; a count alone is a bin count."""
    encoder, cells = declared(name, tmp_path)
    encoder.fit(cells)

    assert encoder.num_classes is None or encoder.num_classes > 0
    for spelled in (encoder.class_names, encoder.class_values):
        if spelled is not None:
            assert len(spelled) == encoder.num_classes


@pytest.mark.parametrize("name", sorted(DECLARED))
def test_the_geometry_is_a_geometry(name: str, tmp_path: Path) -> None:
    encoder, _ = declared(name, tmp_path)

    assert isinstance(encoder.geometry, Geometry)
