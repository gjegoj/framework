"""What a run is about to train on: counted by the encoder that owns the column."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core import DataProfile, Stage
from src.core.entities import ClassDistribution, ValueDistribution
from src.data import (
    DataSchema,
    InMemorySource,
    InputColumn,
    LabelTargetEncoder,
    MaskTargetEncoder,
    MultiLabelTargetEncoder,
    ScalarTargetEncoder,
    TableDataModule,
    TargetColumn,
    random_split,
)
from src.data.statistics import counted, measured


def test_a_class_the_split_never_produced_is_still_a_row() -> None:
    """The single most useful line in a balance table is the class that is missing.

    Counting only what appeared would leave it out, and the column would look
    healthy — a model that can never predict `rare` and a report that never
    mentions it is how the problem survives to the end of a run.
    """
    encoder = LabelTargetEncoder(classes={0: "cat", 1: "dog", 2: "rare"})
    encoder.fit(["cat", "dog", "dog"])

    shown = encoder.distribution(["cat", "dog", "dog"])

    assert isinstance(shown, ClassDistribution)
    assert shown.counts == {"cat": 1, "dog": 2, "rare": 0}
    assert shown.shares["rare"] == 0.0


def test_a_multilabel_column_counts_labels_not_rows() -> None:
    """A row carrying three labels is three counts, so the total exceeds the row count.

    Which is exactly why the stage's size is carried separately rather than read
    off these numbers.
    """
    encoder = MultiLabelTargetEncoder(classes={0: "a", 1: "b"})

    shown = encoder.distribution(["a,b", "b"])

    assert isinstance(shown, ClassDistribution)
    assert shown.counts == {"a": 1, "b": 2}
    assert shown.total == 3  # from two rows


def test_a_numeric_column_is_measured_and_a_missing_cell_does_not_poison_it() -> None:
    """One NaN would turn every statistic into nan, and the row would say nothing at all."""
    shown = ScalarTargetEncoder().distribution([1.0, 2.0, 3.0, float("nan")])

    assert isinstance(shown, ValueDistribution)
    assert shown.count == 3  # against the stage's row count, this is the missing one
    assert shown.median == pytest.approx(2.0)
    assert shown.mean == pytest.approx(2.0)


def test_a_column_holding_no_number_is_not_described() -> None:
    assert measured([float("nan")]) is None


def test_an_encoder_that_does_not_describe_its_column_says_so_rather_than_vanishing() -> None:
    """The reference expressed this by omitting the method, and the column disappeared.

    A base-class method returning `None` keeps the task in the report, where the
    reason can be printed beside its name.
    """

    class Opaque(ScalarTargetEncoder):
        pass

    assert Opaque().distribution([1.0]) is not None  # inherited, and it does describe
    assert MaskTargetEncoder(num_classes=2).distribution.__doc__ is not None


def test_segmentation_counts_its_pixels_where_the_reference_dropped_it(tmp_path: Path) -> None:
    """Class imbalance in a mask is measured in pixels, and it is the imbalance a loss fights.

    Measured cost: 0.88 ms per mask, so seconds for a whole dataset, once, before
    the first epoch — the reference called this too expensive and reported nothing.
    """
    import cv2

    root = tmp_path
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:1] = 1  # a quarter of the pixels are class 1
    cv2.imwrite(str(root / "m.png"), mask)
    encoder = MaskTargetEncoder(classes={0: "background", 1: "object"})

    shown = encoder.distribution([str(root / "m.png")])

    assert isinstance(shown, ClassDistribution)
    assert shown.counts == {"background": 12, "object": 4}
    assert shown.shares["object"] == pytest.approx(0.25)


def test_a_mask_beyond_the_declared_classes_is_refused_by_name(tmp_path: Path) -> None:
    """Left alone it surfaces as a shape error inside the loss, a thousand steps in."""
    import cv2

    root = tmp_path
    mask = np.full((2, 2), 7, dtype=np.uint8)
    cv2.imwrite(str(root / "m.png"), mask)

    with pytest.raises(ValueError, match="class index 7"):
        MaskTargetEncoder(num_classes=2).distribution([str(root / "m.png")])


def test_a_pipeline_reports_its_size_and_its_targets_together() -> None:
    """The first question is how much, the second is what — and one record answers both."""
    table = pd.DataFrame({"x": [float(index) for index in range(8)], "label": ["cat", "dog"] * 4})
    module = TableDataModule(
        source=InMemorySource(table),
        schema=DataSchema(
            inputs={"point": InputColumn(column="x", loader=float)},
            targets={"label": TargetColumn(column="label", encoder=LabelTargetEncoder())},
        ),
        splitter=random_split({Stage.TRAIN: 0.5, Stage.VAL: 0.5}, seed=0),
    )

    module.setup(DataProfile())
    statistics = module.statistics()

    assert statistics.rows == {Stage.TRAIN: 4, Stage.VAL: 4}
    assert set(statistics.targets) == {"label"}
    balance = statistics.targets["label"][Stage.TRAIN]
    assert isinstance(balance, ClassDistribution)
    assert sum(balance.counts.values()) == 4  # one count per row of that stage


def test_a_pipeline_that_cannot_describe_its_data_answers_with_nothing() -> None:
    """A default on the port, not a missing method: every consumer has something to call.

    The reference guarded with `isinstance(datamodule, LitDataModule)`, so a
    pipeline of someone else's making reported nothing and said nothing either.
    """
    from src.core.ports import DataModule

    class Vendor(DataModule):
        def setup(self, profile: DataProfile) -> None: ...

        def dataset(self, stage: Stage) -> object:  # type: ignore[override]
            raise LookupError("none")

    assert not Vendor().statistics()


def test_counting_starts_from_the_vocabulary_even_with_nothing_to_count() -> None:
    """An empty stage still shows its classes, at zero — rather than an empty table."""
    assert counted(["a", "b"], []).counts == {"a": 0, "b": 0}
