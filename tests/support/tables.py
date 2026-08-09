"""An annotation table and a schema over it, held in memory — the pipeline without files.

A test about *splitting*, *combining* or *pre-split sources* is about rows, not about
pixels. These build the rows and the schema that reads them, so such a test never writes
a PNG it will not look at.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import torch

from src.data import DataSchema, InputColumn, LabelTargetEncoder, TargetColumn

if TYPE_CHECKING:
    from torch import Tensor

PATH_COLUMN = "path"
"""Where a row names its image. Never opened by ``load_zeros``."""

LABEL_COLUMN = "label"
"""Where a row names its class."""


def load_zeros(path: object) -> Tensor:
    """An input loader that opens nothing and hands back a fixed vector.

    What the pixels are is never the subject of a table test, and reading a real file
    would make the test depend on one.
    """
    return torch.zeros(3)


def labelled(labels: list[str]) -> pd.DataFrame:
    """One row per label, each naming an image path it shares nothing else with."""
    return pd.DataFrame(
        {
            PATH_COLUMN: [f"{index}.jpg" for index in range(len(labels))],
            LABEL_COLUMN: labels,
        }
    )


def repeated(count: int, label: str = "cat") -> pd.DataFrame:
    """``count`` rows of one label — for tests counting rows rather than classes."""
    return labelled([label] * count)


def label_schema() -> DataSchema:
    """One image input read by ``load_zeros``, and one label target fitted from the rows."""
    return DataSchema(
        inputs={"image": InputColumn(column=PATH_COLUMN, loader=load_zeros)},
        targets={LABEL_COLUMN: TargetColumn(column=LABEL_COLUMN, encoder=LabelTargetEncoder())},
    )
