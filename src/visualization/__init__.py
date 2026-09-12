"""Model outputs drawn over the inputs that made them — a library, not a capability of this framework.

Nothing here imports anything else under ``src``, and nothing but numpy and the standard library from
outside it: a page is built from plain values that describe what a viewer should see, and how they
were arrived at is somebody else's business. That is what makes this directory liftable into a package
of its own, and ``tests/test_layering.py`` holds it to that rather than this docstring.

The framework's side of the seam is ``src/integrations/``, which turns a batch, a step and a run's
tasks into the values below.

This facade publishes the package's whole vocabulary, including the names its own modules use on each
other, and that is deliberate: a library's facade *is* its contract, and what the next caller of it
builds a page from is not something this tree gets to decide. Elsewhere a facade publishes what its
package offers rather than only what this tree happens to read, which is the same argument made once
per package instead of once per name.
"""

from __future__ import annotations

from src.visualization.entities import (
    PREDICTED,
    SIDES,
    TRUTH,
    Classification,
    Classifications,
    Image,
    Label,
    Regression,
    SampleView,
    Score,
    Segmentation,
    SegmentationClass,
    Side,
    Verdict,
)
from src.visualization.html import MAX_DISPLAY_SIDE, HtmlRenderer
from src.visualization.renderers import MAX_CHIP_CHARS

__all__ = [
    "MAX_CHIP_CHARS",
    "MAX_DISPLAY_SIDE",
    "PREDICTED",
    "SIDES",
    "TRUTH",
    "Classification",
    "Classifications",
    "HtmlRenderer",
    "Image",
    "Label",
    "Regression",
    "SampleView",
    "Score",
    "Segmentation",
    "SegmentationClass",
    "Side",
    "Verdict",
]
