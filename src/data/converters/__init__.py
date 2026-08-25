"""Offline converters: the wild annotation layouts into the canonical JSONL.

A converter is run once, by hand, and its output is what a run reads — so the parsing
that a training pipeline would otherwise repeat every epoch happens here, and so does
the validation. A box that leaves its image is clipped *and counted*; one with no area
left refuses by the path of the image it came from. Training never re-reads a wild
format, and an annotation mistake surfaces while a human is still looking.
"""
