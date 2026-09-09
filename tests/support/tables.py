"""An annotation table and a schema over it, held in memory — the pipeline without files.

A test about *splitting*, *combining* or *pre-split sources* is about rows, not about
pixels. These build the rows and the schema that reads them, so such a test never writes
a PNG it will not look at.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import torch

from src.core import Stage
from src.data import (
    DataSchema,
    DeclaredSource,
    InputColumn,
    LabelTargetEncoder,
    TableDataModule,
    TargetColumn,
    random_split,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from torch import Tensor

    from src.core import DatasetFacts, SampleTransform

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
        targets={
            LABEL_COLUMN: TargetColumn(column=LABEL_COLUMN, encoder=LabelTargetEncoder(classes={0: "cat", 1: "dog"}))
        },
    )


def load_point(value: object) -> Tensor:
    """An input loader turning a number into a two-wide vector, ``[x, 1]``.

    Wide enough for a ``LinearHead`` to learn from, small enough that a fit is a blink;
    the constant second lane keeps a one-row batch from being a scalar.
    """
    return torch.tensor([float(value), 1.0])  # type: ignore[arg-type]


def in_memory_pipeline(
    rows: int = 8,
    *,
    input: str = "image",
    loader: Callable[[Any], Any] = load_point,
    transforms: Mapping[Stage, SampleTransform] | None = None,
) -> tuple[TableDataModule, DatasetFacts]:
    """``rows`` numbers under a two-class label, split 50/25/25 and set up; the facts come back with it.

    What a test about training, a callback or a report wants: *a* pipeline whose rows
    are tensors the moment they are read, never files. The input is named ``image`` so
    ``FlattenBackbone`` reads it unchanged; a test with a backbone of its own names the
    input it reads. A test about the pipeline itself still builds its own table.
    """
    table = pd.DataFrame({"x": [float(index) for index in range(rows)], LABEL_COLUMN: ["cat", "dog"] * (rows // 2)})
    module = TableDataModule(
        sources=[DeclaredSource(table)],
        schema=DataSchema(
            inputs={input: InputColumn(column="x", loader=loader)},
            targets={
                LABEL_COLUMN: TargetColumn(
                    column=LABEL_COLUMN, encoder=LabelTargetEncoder(classes={0: "cat", 1: "dog"})
                )
            },
        ),
        splitter=random_split({Stage.TRAIN: 0.5, Stage.VAL: 0.25, Stage.TEST: 0.25}, seed=42),
        transforms=transforms,
    )
    return module, module.setup()
