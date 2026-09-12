"""What a split holds: counted where the cells are labels, measured where they are numbers."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core import ClassDistribution
from src.data.encoders.label import LabelEncoder, MultilabelEncoder
from src.data.encoders.mask import MaskEncoder
from src.data.statistics import counted, measured

CLASSES = {0: "cat", 1: "dog", 2: "bird"}


class TestCounted:
    def test_a_class_the_split_never_produced_is_a_row_rather_than_an_absence(self) -> None:
        """It is the most useful line on the table: a vocabulary naming something the data lacks."""
        assert counted(CLASSES, ["cat", "cat", "dog"]).counts == {"cat": 2, "dog": 1, "bird": 0}

    def test_a_label_outside_the_vocabulary_is_still_counted(self) -> None:
        """The encoders refuse those when they are fitted; a report should not hide the diagnosis."""
        assert counted(CLASSES, ["fish"]).counts["fish"] == 1

    def test_the_shares_are_of_what_was_counted(self) -> None:
        shares = counted(CLASSES, ["cat", "dog"]).shares

        assert shares == {"cat": 0.5, "dog": 0.5, "bird": 0.0}

    def test_an_empty_column_divides_by_nothing_rather_than_failing(self) -> None:
        assert set(counted(CLASSES, []).shares.values()) == {0.0}


class TestMeasured:
    def test_the_five_numbers_and_the_two_beside_them(self) -> None:
        found = measured([1.0, 2.0, 3.0, 4.0])

        assert found is not None
        assert found.deviation == pytest.approx(1.2909944, abs=1e-6), "the sample deviation, not the population one"
        assert (found.count, found.mean, found.minimum, found.maximum) == (4, 2.5, 1.0, 4.0)
        assert (found.q25, found.median, found.q75) == (1.75, 2.5, 3.25)

    def test_a_missing_cell_is_dropped_rather_than_turning_every_number_into_nothing(self) -> None:
        found = measured([1.0, float("nan"), 3.0])

        assert found is not None
        assert found.count == 2
        assert not math.isnan(found.mean)

    def test_a_column_holding_no_number_at_all_has_no_spread(self) -> None:
        assert measured([float("nan")]) is None
        assert measured([]) is None

    def test_one_value_spreads_by_nothing_rather_than_by_a_nan(self) -> None:
        found = measured([2.0])

        assert found is not None
        assert found.deviation == 0.0


class TestEncodersDescribeTheirOwnColumn:
    def test_one_label_per_row(self) -> None:
        described = LabelEncoder(classes=CLASSES).distribution(["cat", "dog", "cat"])

        assert isinstance(described, ClassDistribution)
        assert described.counts == {"cat": 2, "dog": 1, "bird": 0}

    def test_several_labels_per_row_run_the_total_past_the_row_count(self) -> None:
        described = MultilabelEncoder(classes=CLASSES).distribution(["cat,dog", "cat"])

        assert isinstance(described, ClassDistribution)
        assert described.total == 3

    def test_a_mask_is_counted_in_pixels(self, tmp_path: Path) -> None:
        """The imbalance a dense loss spends the run fighting, which no count of rows would show."""
        plane = np.zeros((4, 5), dtype=np.uint8)
        plane[:, :2] = 1
        cv2.imwrite(str(tmp_path / "m.png"), plane)

        described = MaskEncoder(classes=CLASSES).distribution([str(tmp_path / "m.png")])

        assert isinstance(described, ClassDistribution)
        assert described.counts == {"cat": 12, "dog": 8, "bird": 0}

    def test_a_mask_holding_a_class_the_task_never_declared_is_named(self, tmp_path: Path) -> None:
        cv2.imwrite(str(tmp_path / "m.png"), np.full((2, 2), 7, dtype=np.uint8))

        with pytest.raises(ValueError, match="class index 7"):
            MaskEncoder(classes=CLASSES).distribution([str(tmp_path / "m.png")])


class TestTheColumnIsReadTheWayTheRunReadsIt:
    def test_a_label_column_written_as_indices_counts_under_the_declared_names(self) -> None:
        """A vocabulary accepts a name and the index behind it, so such a column trains and validates
        exactly like one written in words. Counting the raw cell reports every declared class as one
        the data never shows — on a perfectly balanced column."""
        described = LabelEncoder(classes=CLASSES).distribution([0, 1, 1, 0, 1])

        assert isinstance(described, ClassDistribution)
        assert described.counts == {"cat": 2, "dog": 3, "bird": 0}

    def test_a_multilabel_cell_written_as_indices_reads_the_same_way(self) -> None:
        described = MultilabelEncoder(classes=CLASSES).distribution(["0,1", "1"])

        assert isinstance(described, ClassDistribution)
        assert described.counts == {"cat": 1, "dog": 2, "bird": 0}

    def test_a_cell_outside_the_vocabulary_keeps_its_own_spelling(self) -> None:
        """The encoders refuse those when they are fitted; a report should not be what hides it."""
        described = LabelEncoder(classes=CLASSES).distribution(["fish"])

        assert isinstance(described, ClassDistribution)
        assert described.counts["fish"] == 1

    @pytest.mark.parametrize("cells", [["a", "b"], [1.0, None, 2.0], [None]], ids=repr)
    def test_a_cell_that_is_not_a_number_is_a_missing_cell_and_not_a_dead_run(self, cells: list[object]) -> None:
        """This runs from a display, before the first epoch: an unnamed conversion error there ends a
        run over a stray `n/a`, where the count against the row count already says one was missing."""
        found = measured(cells)

        assert found is None or found.count == sum(isinstance(one, float) for one in cells)
