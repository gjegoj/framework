"""Demo sample views: the approval page of spec D8 and the renderer's own fixture.

What the user approved is what the tests assert against, so the page they
clicked through and the page the tests read are built by the same function.

Every label kind appears, plus the cases that have broken renderers before: a
task named with an underscore, a class named with markup, a URL source beside a
local path, and a sample carrying an image *and* a caption.

Every cell carries a source, because a cell without one reads as a fault on a
page whose whole job is to be judged by eye. That a sourceless image simply
draws no pill is pinned by a unit test instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.visualization import (
    Classification,
    Classifications,
    Image,
    Media,
    Regression,
    SampleView,
    Score,
    Segmentation,
    SegmentationClass,
    Text,
    Verdict,
)

_SIDE = 96


def _image(seed: int) -> np.ndarray:
    """A deterministic gradient plus noise, so the page needs no files on disk."""
    rng = np.random.default_rng(seed)
    gradient = np.linspace(30, 210, _SIDE)[None, :, None]
    tint = np.array([(seed * 37) % 70, (seed * 61) % 70, (seed * 89) % 70])
    noise = rng.integers(0, 35, size=(_SIDE, _SIDE, 3))
    pixels: np.ndarray = np.clip(gradient + tint + noise, 0, 255).astype(np.uint8)
    return pixels


def _circle(center: tuple[int, int], radius: int) -> np.ndarray:
    rows, columns = np.mgrid[0:_SIDE, 0:_SIDE]
    inside: np.ndarray = (rows - center[0]) ** 2 + (columns - center[1]) ** 2 <= radius**2
    return inside


def _shown(seed: int, source: str | None = None) -> dict[str, Media]:
    return {"image": Image(pixels=_image(seed), source=source)}


def _wide(seed: int) -> np.ndarray:
    """A 2:1 picture — the shape that showed the cell had no coordinate system.

    With the picture cropped and its masks stretched, a mask over the left third
    of a wide image was drawn across the middle of a square cell.
    """
    return _image(seed)[:, : _SIDE // 2 * 2][: _SIDE // 2]


def _segmented(seed: int, offset: int, miou: float, source: str) -> SampleView:
    """A segmentation cell: the same two classes, predicted a little off, scored by mIoU."""
    view = SampleView(media=_shown(seed, source))
    view.fields[("parts", "gt")] = Segmentation(
        classes=(SegmentationClass("body", _circle((48, 40), 26)), SegmentationClass("head", _circle((24, 30), 12)))
    )
    view.fields[("parts", "pred")] = Segmentation(
        classes=(
            SegmentationClass("body", _circle((48, 40 + offset), 26)),
            SegmentationClass("head", _circle((24 + offset // 2, 30 + offset), 13)),
        )
    )
    view.verdicts["parts"] = Verdict(scores=(Score(name="iou", value=miou),))
    return view


def _regressed(seed: int, truth: float, predicted: float, source: str | None = None) -> SampleView:
    """A regression cell: two chips side by side, and how far apart they are as the score."""
    view = SampleView(media=_shown(seed, source))
    view.fields[("age", "gt")] = Regression(value=truth)
    view.fields[("age", "pred")] = Regression(value=predicted)
    view.verdicts["age"] = Verdict(scores=(Score(name="mae", value=abs(predicted - truth)),))
    return view


def demo_views() -> list[SampleView]:
    """Every label kind, every verdict kind, and every case that has bitten a renderer."""
    correct = SampleView(media=_shown(0, "https://example.com/pets/cat_0001.jpg"))
    correct.fields[("species", "gt")] = Classification(label="cat")
    correct.fields[("species", "pred")] = Classification(label="cat", confidence=0.94)
    correct.verdicts["species"] = Verdict(correct=True)

    wrong = SampleView(media=_shown(1, "/data/pets/images/dog_0042.jpg"))
    wrong.fields[("species", "gt")] = Classification(label="dog")
    wrong.fields[("species", "pred")] = Classification(label="cat", confidence=0.51)
    wrong.verdicts["species"] = Verdict(correct=False)
    wrong.fields[("tags", "gt")] = Classifications(
        classifications=(Classification("outdoor"), Classification("close-up"))
    )
    wrong.fields[("tags", "pred")] = Classifications(
        classifications=(
            Classification("outdoor", confidence=0.88),
            Classification("studio", confidence=0.57),
        )
    )
    wrong.verdicts["tags"] = Verdict(correct=False)

    # One task right and one wrong: this is the sample that says what "correct"
    # means at the sample level, and it must read as a mistake.
    partly = SampleView(media=_shown(6, "/data/pets/images/dog_0043.jpg"))
    partly.fields[("species", "gt")] = Classification(label="dog")
    partly.fields[("species", "pred")] = Classification(label="dog", confidence=0.79)
    partly.verdicts["species"] = Verdict(correct=True)
    partly.fields[("tags", "gt")] = Classifications(classifications=(Classification("indoor"),))
    partly.fields[("tags", "pred")] = Classifications(
        classifications=(Classification("indoor", confidence=0.72), Classification("studio", confidence=0.55))
    )
    partly.verdicts["tags"] = Verdict(correct=False)

    # The escaping proof: a class named with markup must render as text. It looks
    # alarming on purpose — a cell that renders it as *markup* would look calm.
    hostile = SampleView(media=_shown(4, "/data/pets/images/cat_0007.jpg"))
    hostile.fields[("my_task", "gt")] = Classification(label="ok")
    hostile.fields[("my_task", "pred")] = Classification(label="<script>alert(1)</script>", confidence=0.5)
    hostile.verdicts["my_task"] = Verdict(correct=False)

    multimodal = SampleView(
        media={
            "image": Image(pixels=_image(5), source="/data/captioned/0007.jpg"),
            "caption": Text(text="a tabby cat asleep on a sunlit windowsill"),
        }
    )
    multimodal.fields[("match", "gt")] = Classification(label="match")
    multimodal.fields[("match", "pred")] = Classification(label="match", confidence=0.81)
    multimodal.verdicts["match"] = Verdict(correct=True)

    # A wide picture with a mask that must land on its own left third. Square
    # samples hid the defect entirely, which is why one is kept here for good.
    wide = SampleView(media={"image": Image(pixels=_wide(11), source="/data/wide/0011.jpg")})
    stripe = np.zeros((_SIDE // 2, _SIDE), dtype=bool)
    stripe[:, : _SIDE // 3] = True
    wide.fields[("parts", "gt")] = Segmentation(classes=(SegmentationClass("body", stripe),))
    wide.fields[("parts", "pred")] = Segmentation(classes=(SegmentationClass("body", np.roll(stripe, 6, axis=1)),))
    # Two measures of one task: the case that made `scores` plural, and the one that
    # gives this page two sliders under one name.
    wide.verdicts["parts"] = Verdict(scores=(Score(name="iou", value=0.71), Score(name="dice", value=0.83)))

    return [
        wide,
        correct,
        wrong,
        partly,
        hostile,
        multimodal,
        _regressed(2, truth=4.0, predicted=5.2, source="/data/pets/images/cat_0100.jpg"),
        _regressed(7, truth=3.0, predicted=3.15, source="/data/pets/images/cat_0101.jpg"),
        _segmented(3, offset=12, miou=0.62, source="/data/parts/masks/0003.png"),
        _segmented(8, offset=26, miou=0.31, source="/data/parts/masks/0008.png"),
        _segmented(9, offset=3, miou=0.88, source="https://example.com/parts/0009.png"),
    ]


def write_demo(path: str | Path) -> Path:
    """Render the demo to ``path`` — the approval gate's page."""
    from src.visualization.html import HtmlRenderer

    destination = Path(path)
    destination.write_text(HtmlRenderer().render(demo_views(), title="samples/val (demo)"), encoding="utf-8")
    return destination
