"""Detection objects: a cell of ``{"box": [x1, y1, x2, y2], "class": name}`` mappings."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar, override

import numpy as np
import torch

from src.core.entities import Distribution, Instances
from src.core.taxonomy import Geometry
from src.core.vocabulary import ordered_names
from src.data.encoders.base import TargetEncoder
from src.data.registry import target_encoder_registry
from src.data.statistics import counted

log = logging.getLogger(__name__)


@target_encoder_registry.register("boxes")
class BoxesTargetEncoder(TargetEncoder):
    """Detection objects: a cell of ``{"box": [x1, y1, x2, y2], "class": name}`` mappings.

    ``load`` accepts the parsed list a JSON table holds or the JSON string a CSV cell holds,
    and returns ``(float32 [N, 4] xyxy pixels, list of names)`` — coordinates in the pixels
    of the image as loaded, so only the pipeline's geometry ever moves them. ``encode`` runs
    after the transforms and is this target's tensor boundary: a ragged value cannot wait for
    stacking, so the result is a per-sample ``Instances`` with ``sample_index`` zeros —
    collation renumbers. A malformed cell is refused showing what it held.

    Parameters:
        classes (Mapping[int, str] | None): Declared vocabulary, index to name. Learned from
            the training cells when absent — the annotations carry the names.
    """

    geometry: ClassVar[Geometry] = Geometry.BOXES

    BOX: ClassVar[str] = "box"
    CLASS: ClassVar[str] = "class"
    """The canonical object fields, spelled once: the converters write what this reads."""

    def __init__(self, classes: Mapping[int, str] | None = None) -> None:
        names = ordered_names(classes) if classes is not None else None
        self._declared = names is not None
        self._index: dict[str, int] | None = (
            {name: position for position, name in enumerate(names)} if names is not None else None
        )

    def fit(self, values: Iterable[Any]) -> None:
        seen = {name for value in values for name in self._parsed(value)[1]}
        if self._declared:
            assert self._index is not None
            unknown = sorted(seen - self._index.keys())
            if unknown:
                known = ", ".join(self._index)
                raise LookupError(f"Unknown classes {', '.join(unknown)} in a boxes column. Declared: {known}.")
            return
        self._index = {name: position for position, name in enumerate(sorted(seen))}

    @override
    def load(self, value: Any) -> tuple[np.ndarray, list[str]]:
        """One cell into the pair the transforms receive — parsing here, indexing at encode."""
        return self._parsed(value)

    def encode(self, value: Any) -> Instances:
        if self._index is None:
            raise RuntimeError("BoxesTargetEncoder is not fitted; call fit(train_values) first.")
        boxes, names = value
        unknown = sorted({name for name in names if name not in self._index})
        if unknown:
            known = ", ".join(self._index)
            raise LookupError(f"Unknown classes {', '.join(unknown)} in a boxes target. Known classes: {known}.")
        return Instances(
            boxes=torch.as_tensor(np.asarray(boxes, dtype=np.float32).reshape(len(names), 4)),
            labels=torch.as_tensor([self._index[name] for name in names], dtype=torch.int64),
            sample_index=torch.zeros(len(names), dtype=torch.int64),
        )

    @property
    def num_classes(self) -> int | None:
        return len(self._index) if self._index is not None else None

    @property
    def class_names(self) -> list[str] | None:
        return list(self._index) if self._index is not None else None

    @override
    def distribution(self, values: Iterable[Any]) -> Distribution | None:
        """Boxes per class across the split — seeded, so a class no image shows still reports."""
        return counted(self.class_names, (name for value in values for name in self._parsed(value)[1]))

    def _parsed(self, value: Any) -> tuple[np.ndarray, list[str]]:
        """The cell in either form (list or JSON string), refused by content when it is not one."""
        listed = json.loads(value) if isinstance(value, str) else value
        if not isinstance(listed, list):
            # TypeError for the wrong *kind* of value, ValueError below for the wrong
            # contents — the split ``require_tensor`` already makes for task outputs.
            raise TypeError(f"A boxes cell holds a list of objects, got {type(listed).__name__}: {listed!r:.120}.")
        boxes: list[list[float]] = []
        names: list[str] = []
        for entry in listed:
            if not isinstance(entry, Mapping) or self.BOX not in entry or self.CLASS not in entry:
                raise ValueError(f"A boxes object needs '{self.BOX}' and '{self.CLASS}', got {entry!r:.120}.")
            corners = [float(corner) for corner in entry[self.BOX]]
            if len(corners) != 4:
                raise ValueError(f"A '{self.BOX}' holds [x1, y1, x2, y2], got {entry[self.BOX]!r:.120}.")
            boxes.append(corners)
            names.append(str(entry[self.CLASS]))
        return np.asarray(boxes, dtype=np.float32).reshape(len(names), 4), names
