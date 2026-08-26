"""The native YOLO layout (``data.yaml`` + ``images/`` + ``labels/*.txt``) into ``.jsonl``.

Run once per dataset::

    uv run python -m src.data.converters.yolo --data path/data.yaml --into data/pets/

One file per stage the descriptor declares. Normalised ``cxcywh`` becomes pixels (which
needs each image's size), and class indices become the descriptor's names.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from PIL import Image

from src.core.taxonomy import Stage
from src.data.converters.annotations import (
    ConversionReport,
    annotation_object,
    annotation_row,
    clipped_box,
    write_annotations,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

IMAGE_SUFFIXES = frozenset(
    {".avif", ".bmp", ".dng", ".heic", ".heif", ".jp2", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
)
"""What counts as a picture in a stage directory.

Copied from the reference tool's ``IMG_FORMATS`` rather than imported — ultralytics is
AGPL-3.0 and quarantined behind the vendor adapters; these offline converters must not
become a second doorway into it. Anything else beside the pictures (.DS_Store, the
reference tool's own ``*.cache``) is detritus, not a lost annotation, and is skipped:
measured, opening .DS_Store as a picture raised ``UnidentifiedImageError`` naming
nothing a user could act on.
"""


def convert(data_yaml: Path | str, *, into: Path | str) -> ConversionReport:
    """Convert every stage the descriptor declares; return what the pass did.

    Parameters:
        data_yaml (Path | str): The YOLO descriptor — its ``path`` root, its per-stage
            image directories, and its ``names`` table.
        into (Path | str): Directory the ``<stage>.jsonl`` files are written to.
    """
    descriptor_path = Path(data_yaml)
    descriptor: dict[str, Any] = yaml.safe_load(descriptor_path.read_text(encoding="utf-8"))
    # A relative `path:` is relative to the descriptor, as the reference tool resolves it
    # — measured, resolving against the CWD converted 0 images from a sibling directory
    # and wrote an empty file without one refusal. pathlib keeps an absolute right
    # operand as-is, so both spellings work in one expression.
    root = descriptor_path.parent / str(descriptor.get("path", "."))
    names = {int(index): str(name) for index, name in dict(descriptor["names"]).items()}
    report = ConversionReport()
    # The descriptor's stage keys are the framework's own names; one it does not declare
    # is simply absent, as in YOLO practice.
    for stage in Stage:
        declared = descriptor.get(stage)
        if not declared:
            continue
        records = [_record(image, root, names, report) for image in _images_of(root / str(declared))]
        if not records:
            raise ValueError(
                f"Stage '{stage}' declares '{declared}' in {descriptor_path.name}, but no images were "
                f"found under {root / str(declared)}. A declared stage with nothing in it is a "
                f"resolution mistake (a wrong 'path:', a moved directory), not an empty split."
            )
        write_annotations(records, Path(into) / f"{stage}.jsonl")
    return report


def _images_of(directory: Path) -> Iterator[Path]:
    """Every image the stage holds, and every image its labels claim.

    The union rather than the picture files alone: a label file whose image was deleted
    is annotations going missing, and the refusal that names the absent image — at the
    path in *this* directory, where it should have been — is more useful than a
    converted dataset that quietly lost rows. The stem index is built once for the
    directory: measured, rebuilding it per label file was 70 s where one build is
    0.05 s at 20k images x 20k labels.
    """
    pictures = sorted(path for path in directory.glob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    stems = {path.stem: path for path in pictures}
    labelled = (
        stems.get(label.stem, directory / f"{label.stem}.jpg") for label in _labels_dir(directory).glob("*.txt")
    )
    yield from sorted(set(pictures) | set(labelled))


def _labels_dir(images: Path) -> Path:
    """YOLO's convention: the *last* ``images`` path segment becomes ``labels``.

    The last, not every one — the reference implementation replaces the final
    ``/images/`` occurrence (read from ``ultralytics.data.utils.img2label_paths``, not
    imported: ultralytics is AGPL-3.0 and quarantined behind the vendor adapters).
    Measured on the every-segment spelling: a dataset under a parent directory named
    ``images`` had its labels looked up in a directory that does not exist, and every
    row converted as a negative. No ``images`` segment at all means the labels sit
    beside the pictures, which is the same reference behaviour.
    """
    parts = list(images.parts)
    for position in range(len(parts) - 1, -1, -1):
        if parts[position] == "images":
            parts[position] = "labels"
            break
    return Path(*parts)


def _label_of(image: Path) -> Path:
    """The label file of one image, by the same convention, with a ``.txt`` suffix."""
    return _labels_dir(image.parent) / f"{image.stem}.txt"


def _parsed_line(
    line: str, number: int, label: Path, names: dict[int, str]
) -> tuple[str, tuple[float, float, float, float]]:
    """One label line into its class name and normalised cxcywh — refused with an address.

    The refusal names the file, the 1-based line and the expected form; the encoder-side
    refusals cannot, being handed values rather than rows.
    """
    fields = line.split()
    if len(fields) != 5:
        raise ValueError(
            f"{label}:{number}: a YOLO detection label line is 'class cx cy w h' (5 normalised "
            f"fields), got {len(fields)}: {line!r:.80}. A longer line is usually a segmentation "
            f"polygon; this converter reads detection labels."
        )
    try:
        index = int(fields[0])
        centre_x, centre_y, box_width, box_height = (float(value) for value in fields[1:])
    except ValueError as error:
        raise ValueError(
            f"{label}:{number}: a YOLO detection label line is 'class cx cy w h', got {line!r:.80}."
        ) from error
    if index not in names:
        declared = ", ".join(f"{position}: {name}" for position, name in sorted(names.items()))
        raise ValueError(f"{label}:{number}: class index {index} is not in the descriptor's names ({declared}).")
    return names[index], (centre_x, centre_y, box_width, box_height)


def _record(image: Path, root: Path, names: dict[int, str], report: ConversionReport) -> dict[str, Any]:
    """One annotation row: the image's path relative to the root, its objects in pixels."""
    if not image.exists():
        raise FileNotFoundError(f"Image file not found: {image} (its label file says it should be there).")
    relative = str(image.relative_to(root))
    with Image.open(image) as opened:  # header only — the pixels are never decoded
        width, height = opened.size
    objects = []
    label = _label_of(image)
    lines = label.read_text(encoding="utf-8").splitlines() if label.exists() else []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        name, (centre_x, centre_y, box_width, box_height) = _parsed_line(line, number, label, names)
        half_width, half_height = box_width / 2, box_height / 2
        corners = (
            (centre_x - half_width) * width,
            (centre_y - half_height) * height,
            (centre_x + half_width) * width,
            (centre_y + half_height) * height,
        )
        bounded = clipped_box(corners, width=width, height=height, image=relative, report=report)
        objects.append(annotation_object(bounded, name))
        report.objects += 1
    report.images += 1
    return annotation_row(relative, objects)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="path to the YOLO data.yaml descriptor")
    parser.add_argument("--into", required=True, help="directory for the annotation files, one per stage")
    arguments = parser.parse_args()
    print(convert(arguments.data, into=arguments.into).summary())


if __name__ == "__main__":
    main()
