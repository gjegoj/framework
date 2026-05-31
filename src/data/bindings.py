"""TargetBinding: links a task to the data column and codec that produce its target.

A data-layer DTO (it depends on ``TargetCodec``, a data port), kept in its own
file rather than buried in ``dataset.py`` so it is discoverable next to the other
single-responsibility data modules (``sources``/``codecs``/``transforms``).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data.codecs import TargetCodec


@dataclass(frozen=True)
class TargetBinding:
    """Binds a task to the data column and codec that produce its target.

    Parameters:
        name (str): Task name; also the key under which the target is stored.
        column (str): Source column in the DataFrame.
        codec (TargetCodec): Codec that decodes the raw column value.
    """

    name: str
    column: str
    codec: TargetCodec
