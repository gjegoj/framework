"""An `export` declaration becomes the formats this run ships its trained model in."""

from __future__ import annotations

from collections.abc import Sequence

from src.config import ComponentConfig
from src.config.instantiate import instantiate
from src.export.base import Exporter
from src.export.registry import exporter_registry


def build_exporters(declared: Sequence[ComponentConfig]) -> list[Exporter]:
    """Every format this run writes, in the order it declared them."""
    built = [_one(component) for component in declared]
    _refuse_two_formats_writing_one_file(declared, built)
    return built


def _one(declared: ComponentConfig) -> Exporter:
    built = instantiate(declared, exporter_registry)
    if not isinstance(built, Exporter):
        raise TypeError(
            f"{declared.spelled!r} built {type(built).__name__}, which ships nothing: an exporter writes a "
            "graph to a file and reads that file back, so what was written can be proven to be the model."
        )
    return built


def _refuse_two_formats_writing_one_file(declared: Sequence[ComponentConfig], built: Sequence[Exporter]) -> None:
    """Two declarations landing on one path: the second would replace the first, and the run would ship one.

    Asked of what was built rather than of what was written, because that is the only place the answer
    is. The reference implementation compared the declarations, so a registry name beside a ``_target_``
    naming the same class counted as two formats and wrote one file — with whichever settings came last.
    """
    written: dict[str, str] = {}
    for component, one in zip(declared, built, strict=True):
        if one.suffix in written:
            raise ValueError(
                f"{written[one.suffix]!r} and {component.spelled!r} both write this run's artifact under "
                f"'.{one.suffix}', so one would replace the other. Declare each format once."
            )
        written[one.suffix] = component.spelled
