"""A COCO instances export into the ``.jsonl`` annotation format.

Run once per export::

    uv run python -m src.data.converters.coco \\
        --annotations instances.json --images images/ --into data/pets/

Category ids become names, ``[x, y, w, h]`` becomes ``[x1, y1, x2, y2]``, crowd regions
are dropped and counted. One file, ``annotations.jsonl``: the run's own splitters divide it.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.data.converters.annotations import (
    ConversionReport,
    annotation_object,
    annotation_row,
    clipped_box,
    write_annotations,
)


def convert(annotations: Path | str, *, images: Path | str, into: Path | str) -> ConversionReport:
    """Convert one COCO instances file; return what the pass did.

    Parameters:
        annotations (Path | str): The COCO instances json — its ``images``,
            ``categories`` and ``annotations`` tables.
        images (Path | str): Directory the ``file_name`` entries are relative to.
        into (Path | str): Directory ``annotations.jsonl`` is written to.
    """
    export: dict[str, Any] = json.loads(Path(annotations).read_text(encoding="utf-8"))
    categories = {int(entry["id"]): str(entry["name"]) for entry in export["categories"]}
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in export["annotations"]:
        by_image[int(annotation["image_id"])].append(annotation)
    known = {int(entry["id"]) for entry in export["images"]}
    orphaned = sorted(set(by_image) - known)
    if orphaned:
        shown = ", ".join(str(image_id) for image_id in orphaned[:5])
        raise ValueError(
            f"{len(orphaned)} annotation image id(s) are missing from the export's 'images' table "
            f"(first: {shown}). The export is inconsistent, and converting it would silently drop "
            f"those annotations."
        )
    report = ConversionReport()
    records = [_record(entry, by_image, categories, Path(images), report) for entry in export["images"]]
    write_annotations(records, Path(into) / "annotations.jsonl")
    return report


def _record(
    entry: dict[str, Any],
    by_image: dict[int, list[dict[str, Any]]],
    categories: dict[int, str],
    images: Path,
    report: ConversionReport,
) -> dict[str, Any]:
    """One annotation row: the image the export names, and the objects it keeps."""
    file_name = str(entry["file_name"])
    if not (images / file_name).exists():
        raise FileNotFoundError(f"Image file not found: {images / file_name} (the export lists it).")
    width, height = int(entry["width"]), int(entry["height"])
    objects = []
    for annotation in by_image[int(entry["id"])]:
        if annotation.get("iscrowd"):
            report.dropped_crowd += 1
            continue
        x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
        bounded = clipped_box(
            (x, y, x + box_width, y + box_height),
            width=width,
            height=height,
            image=file_name,
            report=report,
        )
        objects.append(annotation_object(bounded, categories[int(annotation["category_id"])]))
        report.objects += 1
    report.images += 1
    return annotation_row(file_name, objects)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, help="path to the COCO instances json")
    parser.add_argument("--images", required=True, help="directory the file_name entries are relative to")
    parser.add_argument("--into", required=True, help="directory for annotations.jsonl")
    arguments = parser.parse_args()
    print(convert(arguments.annotations, images=arguments.images, into=arguments.into).summary())


if __name__ == "__main__":
    main()
