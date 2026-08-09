"""``TableDataset`` and ``collate_samples``: from table rows to ``Sample`` and ``Batch``."""

from __future__ import annotations

import pandas as pd
import pytest
import torch
from torch import Tensor

from src.core import Batch, Sample
from src.data import (
    DataSchema,
    InputColumn,
    LabelTargetEncoder,
    ScalarTargetEncoder,
    TableDataset,
    TargetColumn,
    collate_samples,
)
from tests.support.narrowing import tensor


def load_fake_image(path: object) -> Tensor:
    # Deterministic stand-in for image I/O: pixel value = path length.
    return torch.full((3,), float(len(str(path))))


def make_schema() -> DataSchema:
    label_encoder = LabelTargetEncoder()
    label_encoder.fit(pd.Series(["cat", "dog"]))
    return DataSchema(
        inputs={"image": InputColumn(column="path", loader=load_fake_image)},
        targets={
            "label": TargetColumn(column="label", encoder=label_encoder),
            "score": TargetColumn(column="score", encoder=ScalarTargetEncoder()),
        },
    )


def make_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "path": ["a.jpg", "bb.jpg"],
            "label": ["cat", "dog"],
            "score": [1.0, 2.0],
        }
    )


def test_dataset_builds_a_sample_from_a_row() -> None:
    dataset = TableDataset(make_table(), make_schema())

    sample = dataset[0]

    assert isinstance(sample, Sample)
    assert torch.equal(sample.inputs["image"], torch.full((3,), 5.0))
    assert sample.targets["label"] == 0
    assert sample.targets["score"] == pytest.approx(1.0)
    assert sample.meta["row"] == 0


def test_a_readable_cell_travels_with_the_sample_and_an_array_does_not() -> None:
    """A string names its content or is its content; the tensor built from it already went to the model.

    This is the whole provenance path: a report that wants to link a file, or to
    print the caption a text encoder tokenized away, has nowhere else to read it.
    """
    dataset = TableDataset(make_table(), make_schema())

    assert dataset[0].meta["cells"] == {"image": "a.jpg"}


def test_a_non_string_cell_is_left_behind() -> None:
    """Carrying an array here would duplicate the batch into every sample's metadata."""
    table = make_table().assign(path=[[1.0, 2.0], [3.0, 4.0]])
    schema = DataSchema(
        inputs={"image": InputColumn(column="path", loader=lambda cell: torch.tensor(cell))},
        targets={"score": TargetColumn(column="score", encoder=ScalarTargetEncoder())},
    )

    assert TableDataset(table, schema)[0].meta["cells"] == {}


def test_a_batch_reads_its_cells_through_one_typed_accessor() -> None:
    """The key is named and shaped in one place, so no consumer spells it for itself."""
    batch = collate_samples([TableDataset(make_table(), make_schema())[index] for index in (0, 1)])

    assert batch.cells == [{"image": "a.jpg"}, {"image": "bb.jpg"}]


def test_a_batch_with_no_cells_at_all_reads_as_empty_not_as_missing() -> None:
    """A run over in-memory arrays has none; a consumer must not have to guess the default."""
    assert Batch(inputs={}, targets={}).cells == []


def test_a_sample_whose_metadata_disagrees_is_named_rather_than_stripped() -> None:
    """Transposing from the first sample's keys would silently drop what the others carried."""
    samples = [TableDataset(make_table(), make_schema())[index] for index in (0, 1)]
    samples[1].meta["extra"] = "something a transform attached"

    with pytest.raises(ValueError, match="Sample 1 carries metadata keys"):
        collate_samples(samples)


def test_dataset_length_matches_the_table() -> None:
    assert len(TableDataset(make_table(), make_schema())) == 2


def test_dataset_applies_the_transform() -> None:
    def double_image(sample: Sample) -> Sample:
        sample.inputs["image"] = sample.inputs["image"] * 2
        return sample

    dataset = TableDataset(make_table(), make_schema(), transform=double_image)

    assert torch.equal(dataset[0].inputs["image"], torch.full((3,), 10.0))


def test_dataset_fails_fast_when_schema_references_missing_columns() -> None:
    schema = DataSchema(
        inputs={"image": InputColumn(column="missing", loader=load_fake_image)},
        targets={},
    )

    with pytest.raises(ValueError, match="missing"):
        TableDataset(make_table(), schema)


def test_collate_stacks_samples_into_a_batch() -> None:
    dataset = TableDataset(make_table(), make_schema())

    batch = collate_samples([dataset[0], dataset[1]])

    assert isinstance(batch, Batch)
    assert batch.inputs["image"].shape == (2, 3)
    assert tensor(batch.targets["label"]).tolist() == [0, 1]
    assert batch.meta["row"] == [0, 1]


def test_collate_rejects_an_empty_list() -> None:
    with pytest.raises(ValueError, match="empty"):
        collate_samples([])
