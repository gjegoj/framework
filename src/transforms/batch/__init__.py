"""Transforms of a collated batch: what a sample transform cannot do, because it sees one sample.

A different contract from the rest of the package — ``BatchTransform``, bound to the run's tasks —
and a different consumer: a callback installs these into the training module, while the chain a
stage declares is run by the data pipeline. Pure torch; no pixel library reaches here.
"""

from __future__ import annotations

from src.transforms.batch.mix import CutMix, LabelMix, MixUp

__all__ = ["CutMix", "LabelMix", "MixUp"]
