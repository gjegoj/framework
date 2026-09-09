"""Building the export capability from its section: the declared deployment formats."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from src.config.instantiate import instantiate
from src.export.registry import exporter_registry

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.config.experiment import ExporterConfig
    from src.export.exporters import Exporter


def build_exporters(declared: Sequence[ExporterConfig] | None) -> list[Exporter]:
    """The declared deployment formats, in the order they are written, refusing two that would write one file."""
    entries = list(declared or [])
    counted = Counter(entry.target or entry.name for entry in entries)
    repeated = sorted(str(reference) for reference, count in counted.items() if count > 1)
    if repeated:
        raise ValueError(
            f"The export section declares {', '.join(repeated)} more than once, and a second target of "
            "one format would overwrite the first. Declare each format once."
        )
    return [instantiate(entry, exporter_registry) for entry in entries]
