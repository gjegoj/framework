"""Detection objects: a cell of ``{"box": [x1, y1, x2, y2], "class": name}`` mappings."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar, override

import numpy as np
import torch

from src.core.entities import Instances
from src.core.taxonomy import Geometry
from src.data.encoders.base import VocabularyTargetEncoder
from src.data.registry import target_encoder_registry
from src.data.statistics import Distribution, counted

log = logging.getLogger(__name__)


@target_encoder_registry.register("boxes")
class BoxesTargetEncoder(VocabularyTargetEncoder):
    """Detection objects: a cell of ``{"box": [x1, y1, x2, y2], "class": name}`` mappings.

    ``load`` accepts the parsed list a JSON table holds or the JSON string a CSV cell holds,
    and returns ``(float32 [N, 4] xyxy pixels, list of names)`` — coordinates in the pixels
    of the image as loaded, so only the pipeline's geometry ever moves them. ``encode`` runs
    after the transforms and is this target's tensor boundary: a ragged value cannot wait for
    stacking, so the result is a per-sample ``Instances`` with ``sample_index`` zeros —
    collation renumbers. A malformed object, a box without area or off the origin, and an
    unknown class are refused showing what the cell held — at setup, over every split.

    Parameters:
        classes (Mapping[int, str]): The vocabulary, index to name.
    """

    geometry: ClassVar[Geometry] = Geometry.BOXES

    BOX: ClassVar[str] = "box"
    CLASS: ClassVar[str] = "class"
    """The canonical object fields, spelled once: the converters write what this reads."""

    @override
    def validate(self, values: Iterable[Any]) -> None:
        """Every cell parsed — geometry and names alike — so a bad row dies at setup, shown."""
        self._refuse_unknown({name for value in values for name in self._parsed(value)[1]}, "a boxes column")

    @override
    def load(self, value: Any) -> tuple[np.ndarray, list[str]]:
        """One cell into the pair the transforms receive — parsing here, indexing at encode."""
        return self._parsed(value)

    @override
    def encode(self, value: Any) -> Instances:
        boxes, names = value
        self._refuse_unknown(set(names), "a boxes target")
        return Instances(
            boxes=torch.as_tensor(np.asarray(boxes, dtype=np.float32)),
            labels=torch.as_tensor([self._positions[name] for name in names], dtype=torch.int64),
            sample_index=torch.zeros(len(names), dtype=torch.int64),
        )

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
            boxes.append(self._corners(entry))
            names.append(str(entry[self.CLASS]))
        # `[]` is shape (0,); a negative is (0, 4).
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), names

    def _corners(self, entry: Mapping[str, Any]) -> list[float]:
        """Four numbers with area, at or past the origin — what needs no image to be checked.

        Whether the box also fits *inside* its image only the loaded image can say; the transform
        seam refuses that, naming the row. Measured on albumentationsx 2.3.7: left to the
        pipeline, either mistake dies inside the first epoch, in normalised coordinates, naming
        no row.
        """
        box = entry[self.BOX]
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(f"A '{self.BOX}' holds [x1, y1, x2, y2], got {box!r:.120}.")
        try:
            x1, y1, x2, y2 = (float(corner) for corner in box)
        except (TypeError, ValueError):
            raise ValueError(f"A '{self.BOX}' holds four numbers [x1, y1, x2, y2], got {box!r:.120}.") from None
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"A '{self.BOX}' holds pixel corners with 0 <= x1 < x2 and 0 <= y1 < y2, got {entry!r:.120}."
            )
        return [x1, y1, x2, y2]

    def _refuse_unknown(self, names: set[str], where: str) -> None:
        unknown = sorted(names - self._positions.keys())
        if unknown:
            raise LookupError(f"Unknown classes {', '.join(unknown)} in {where}. Declared: {', '.join(self._names)}.")
