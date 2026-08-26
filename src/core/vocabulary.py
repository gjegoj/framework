"""The one rule a declared class vocabulary follows, wherever it is declared."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


def ordered_names(classes: Mapping[int, str]) -> list[str]:
    """Index-keyed names as the list their order means; loud on a broken range.

    A declared vocabulary is the index space the model's outputs live in: a gap shifts every
    class above it, a repeated name makes two classes indistinguishable in every report, and
    both fail silently. One rule and one message for both places a vocabulary is declared.

    Raises:
        ValueError: On a gap in the indices, or on a duplicated name.
    """
    missing = sorted(set(range(len(classes))) - set(classes))
    if missing:
        raise ValueError(
            f"Class indices must be exactly 0..{len(classes) - 1}; missing: {', '.join(map(str, missing))}."
        )
    names = [classes[index] for index in range(len(classes))]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise ValueError(f"Class names are duplicated: {', '.join(duplicated)}.")
    return names


__all__ = ["ordered_names"]
