"""The table data module: rows → named splits → fitted encoders → per-split datasets, every step out loud."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar

import pandas as pd
import pytest

from src.core import Sample, require_tensor
from src.data import StandardPreprocessor, TableSource
from src.data.encoders import LinearBinsEncoder
from src.data.split import Split
from tests.data.conftest import ModuleFactory, PreprocessorFactory, pipeline, prepared


def test_setup_then_fit_yields_prepared_samples_from_every_split(
    make_module: ModuleFactory, make_preprocessor: PreprocessorFactory
) -> None:
    binned = make_preprocessor(targets={"age": LinearBinsEncoder(bins=3)})
    module = prepared(make_module(preprocessor=binned, targets={"age": "age"}), ("train", "val", "test"))

    sample = module.dataset("val")[0]

    assert module.info.splits == ("train", "val", "test")
    assert module.info.targets["age"].num_classes == 3
    assert require_tensor(sample.inputs["image"], name="image").shape == (3, 4, 4)
    assert sample.metadata["row"] == 0 and sample.metadata["cells"] == {"image": module.dataset("val").table["path"][0]}


def test_each_split_runs_the_pipeline_its_own_stage_declared(
    make_module: ModuleFactory, make_preprocessor: PreprocessorFactory
) -> None:
    """Augmentation is one stage's business: the train pipeline runs for train rows and no others."""
    seen: list[str] = []
    preprocessor = make_preprocessor()
    prepare = pipeline(preprocessor)

    def augmented(sample: Sample) -> Sample:
        seen.append("train")
        return prepare(sample)

    module = prepared(
        make_module(
            preprocessor=preprocessor,
            split=Split({"train": 0.5, "val": 0.5}),
            transforms={"train": augmented, "val": prepare},
        ),
        ("train", "val"),
    )
    _ = module.dataset("train")[0], module.dataset("val")[0]

    assert seen == ["train"]


def test_a_cap_applies_before_dividing(make_module: ModuleFactory) -> None:
    module = prepared(make_module(split=Split({"train": 0.5, "val": 0.5}), max_samples=6), ("train", "val"))

    assert len(module.dataset("train")) + len(module.dataset("val")) == 6


def test_a_split_that_was_not_set_up_is_named(make_module: ModuleFactory) -> None:
    module = prepared(make_module(split=Split({"train": 1.0})), ("train",))

    with pytest.raises(LookupError, match="val"):
        module.dataset("val")


def test_a_column_the_table_lacks_is_named(make_module: ModuleFactory) -> None:
    with pytest.raises(ValueError, match="nope"):
        make_module(inputs={"image": "nope"}, split=Split({"train": 1.0})).setup(("train",))


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param(
            {"source": {"train": pd.DataFrame()}, "split": Split({"train": 1.0})}, id="divided sources plus a split"
        ),
        pytest.param({"split": None}, id="one source and nothing to divide it"),
    ],
)
def test_exactly_one_of_a_split_or_divided_sources(make_module: ModuleFactory, overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="split"):
        make_module(**overrides)


class TestDividedSources:
    def test_are_read_only_for_the_splits_asked_for(self, table: pd.DataFrame, make_module: ModuleFactory) -> None:
        class Counting(TableSource):
            reads: ClassVar[int] = 0

            def read(self) -> pd.DataFrame:
                type(self).reads += 1
                return table

        module = make_module(source={"train": Counting(), "test": Counting()}, split=None)
        module.setup(("test",))

        assert Counting.reads == 1 and module.info.splits == ("test",)

    def test_fitting_validates_the_other_splits_and_names_the_offender(
        self, table: pd.DataFrame, make_module: ModuleFactory
    ) -> None:
        odd = table.copy()
        odd.loc[len(odd) - 1, "species"] = "bird"
        module = make_module(source={"train": odd.head(3), "val": odd.tail(1)}, split=None)
        module.setup(("train", "val"))

        with pytest.raises(LookupError, match=r"val.*species.*bird"):
            module.fit_preprocessing("train")


def test_warm_feeds_the_rows_of_the_named_splits_to_the_preprocessor(
    make_module: ModuleFactory, make_preprocessor: PreprocessorFactory
) -> None:
    class Recording(StandardPreprocessor):
        warmed: ClassVar[list[tuple[str, int]]] = []

        def warm(self, samples: Iterable[Sample], label: str) -> None:
            self.warmed.append((label, sum(1 for _ in samples)))

    recording = Recording(inputs=make_preprocessor().inputs, targets={}, collator=make_preprocessor().collator)
    module = prepared(make_module(preprocessor=recording, targets={}), ("train", "val", "test"))
    module.warm(("train", "val"))

    assert recording.warmed == [("train", 6), ("val", 3)]
