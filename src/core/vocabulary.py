"""The one rule a declared class vocabulary follows, wherever it is declared."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


def ordered_names(classes: Mapping[int, str]) -> list[str]:
    """Index-keyed names as the list their order means; loud on a broken range.

    A declared vocabulary *is* the index space a model's outputs live in, so it has to be
    complete and unambiguous. A gap shifts every class above it by one; a repeated name
    makes two of them indistinguishable in every report the run writes. Both fail in
    silence — the run trains, and reports, against a class space nobody meant.

    Here rather than at either of its callers, because a vocabulary is declared in two
    places: on a task, where pydantic validates the section, and to an encoder built
    directly from Python. Those two carried the same two rules and near-identical
    messages, with a comment on one of them saying it mirrored the other. One rule, one
    message, one test.

    Parameters:
        classes (Mapping[int, str]): The declared vocabulary, index to name.

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
