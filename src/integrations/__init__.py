"""The framework's side of the display library: this run's tensors as values a page can be drawn from.

``src/visualization/`` knows images, labels and colours, and nothing about tasks, batches or torch —
which is what lets it leave this tree. Every translation between the two vocabularies happens here, so
the arrow points this way only and the library never learns what a ``Task`` is. A caller may still
build a page itself — the samples callback holds the renderer and its size knob — but what it hands
the renderer was made here.
"""

from __future__ import annotations

from src.integrations.visualization import Gallery, annotator_for, drawn_input, vocabulary_of

__all__ = ["Gallery", "annotator_for", "drawn_input", "vocabulary_of"]
