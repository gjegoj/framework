"""``scripts/prepare_pet.py``: the claims its output rests on."""

from __future__ import annotations

import importlib.util
import sys
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[3] / "scripts" / "prepare_pet.py"


@cache
def script() -> Any:
    """Load the script as a module; it is not a package, so it is loaded by path.

    Registered in `sys.modules` before it runs: a dataclass resolves its annotations
    through `sys.modules[cls.__module__]`, and a module executed without being
    registered has no entry to resolve against.
    """
    spec = importlib.util.spec_from_file_location("prepare_pet", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def staged(root: Path) -> tuple[Any, Path, Path, Path]:
    """The script and the three empty directories it reads and writes.

    Every test below starts from exactly this, and what makes it different is the files
    it then puts into them.
    """
    images, trimaps, masks = root / "i", root / "t", root / "m"
    for directory in (images, trimaps, masks):
        directory.mkdir()
    return script(), images, trimaps, masks


def test_an_age_follows_its_row_rather_than_its_position() -> None:
    """Keyed by the image's name, so a row keeps its age through a filter or a re-run.

    Position-keyed noise would relabel the whole table the first time a row was
    dropped, and two runs of the same script would disagree about what they trained
    on — which for a *prepared* dataset is worse than having no ages at all.
    """
    prepare = script()

    assert prepare.random_age("Abyssinian_100", 0) == prepare.random_age("Abyssinian_100", 0)
    assert prepare.random_age("Abyssinian_100", 0) != prepare.random_age("Abyssinian_101", 0)
    assert prepare.random_age("Abyssinian_100", 0) != prepare.random_age("Abyssinian_100", 1)


def test_an_age_stays_inside_the_range_it_claims() -> None:
    """The column has to read like data, or it distracts from what is being tested."""
    prepare = script()
    low, high = prepare.AGE_RANGE

    ages = [prepare.random_age(f"breed_{index}", 0) for index in range(500)]

    assert all(low <= age <= high for age in ages)


def test_a_trimap_becomes_zero_based_class_indices() -> None:
    """Oxford stores `{1, 2, 3}`; cross-entropy counts from zero.

    Off by one, every pixel would be one class over and the background class would
    never appear in a target — a segmentation that trains and means nothing.
    """
    prepare = script()

    remapped = prepare.zero_based(np.array([[1, 2, 3]], dtype=np.uint8))

    assert remapped.tolist() == [[0, 1, 2]]
    assert remapped.dtype == np.uint8


def test_the_index_is_read_whole_and_its_header_skipped(tmp_path: Path) -> None:
    """`list.txt` is the entire dataset; `trainval.txt` is half of it, which is what
    the reference read without saying so. The file also opens with commented lines.
    """
    prepare = script()
    (tmp_path / "list.txt").write_text(
        "#Image CLASS-ID SPECIES BREED ID\n#SPECIES: 1:Cat 2:Dog\nAbyssinian_100 1 1 1\nbeagle_10 10 2 2\n"
    )

    rows = prepare.listed(tmp_path)

    assert [(row.name, row.species, row.breed) for row in rows] == [
        ("Abyssinian_100", "cat", "Abyssinian"),
        ("beagle_10", "dog", "beagle"),
    ]


def test_a_row_whose_files_are_missing_is_counted_not_dropped_in_silence(tmp_path: Path) -> None:
    """A preparation script whose output cannot be reconciled with its input is one
    nobody can trust. The reference `continue`d with no count and no reason.
    """
    prepare, images, trimaps, masks = staged(tmp_path)
    rows = [prepare.Listed(name="ghost_1", species="dog")]

    records, skipped = prepare.prepared(rows, images, trimaps, masks, 0)

    assert records == []
    assert skipped == {"missing file": 1}


def test_a_file_that_cannot_be_decoded_is_counted_by_its_own_reason(tmp_path: Path) -> None:
    """Verified with the same OpenCV the framework's `ImageLoader` uses, so "readable
    here" means "readable there" — and a `None` return is checked rather than an
    exception caught and guessed at.
    """
    import cv2

    prepare, images, trimaps, masks = staged(tmp_path)
    (images / "broken_1.jpg").write_bytes(b"not an image")
    cv2.imwrite(str(trimaps / "broken_1.png"), np.ones((2, 2), dtype=np.uint8))
    rows = [prepare.Listed(name="broken_1", species="cat")]

    records, skipped = prepare.prepared(rows, images, trimaps, masks, 0)

    assert records == []
    assert skipped == {"unreadable image": 1}


def test_a_good_row_becomes_a_record_with_its_mask_written(tmp_path: Path) -> None:
    """The whole path, on two real files: a table row out, a zero-based mask on disk."""
    import cv2

    prepare, images, trimaps, masks = staged(tmp_path)
    cv2.imwrite(str(images / "beagle_1.jpg"), np.full((4, 4, 3), 128, dtype=np.uint8))
    cv2.imwrite(str(trimaps / "beagle_1.png"), np.array([[1, 2], [3, 1]], dtype=np.uint8))

    records, skipped = prepare.prepared([prepare.Listed("beagle_1", "dog")], images, trimaps, masks, 0)

    (record,) = records
    assert not skipped
    assert record["species"] == "dog" and record["breed"] == "beagle"
    assert float(record["random_age"]) == pytest.approx(prepare.random_age("beagle_1", 0))
    written = cv2.imread(str(masks / "beagle_1.png"), cv2.IMREAD_GRAYSCALE)
    assert written is not None  # the mask was written, and `imread` says so by not answering None
    assert written.tolist() == [[0, 1], [2, 0]]
