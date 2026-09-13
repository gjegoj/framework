"""What every deployment format must be able to do: write a graph to a file, and read that file back."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import Tensor

from src.export.deployable import DeployableModel

type Runnable = Callable[[tuple[Tensor, ...]], tuple[Tensor, ...]]
"""Runs a written artifact the way that format's own runtime loads it.

Positional, as the graph's own call is: a backend that feeds its inputs by name reads those names out
of the file it has just written, which is also what proves they landed in it.
"""

ABSOLUTE_TOLERANCE = 1e-4
RELATIVE_TOLERANCE = 1e-3
"""How far a written artifact may stand from the model before it is no longer that model.

One home for both numbers, because they are what every format is proven against. A format that drifts
further by construction takes its own through its own constructor, and the declaration is where the
value comes from: the shipped `tensorrt.yaml` says to raise them alongside half precision rather than
carrying a second default, because nobody here has a machine on which to measure what the right one is.
"""

WRITTEN_FROM = 2
"""The fewest rows an artifact can be written from, and therefore the number every run writes from.

An axis of size one is settled into the graph, so two is where a batch can still be generalized out of
an example; more buys nothing, because what a format does to two rows it does to any number of them and
verification asks each artifact at a second size regardless.
"""

BATCH_AXIS = 0
"""Which axis a deployment chooses the size of.

One row is the case that matters: whatever batch a run trained at, what is served is usually a single
sample, and an artifact that could not take one would be useless at exactly the moment it is used.
"""


def beside(destination: Path, suffix: str) -> Path:
    """``destination`` with a suffix added to its name — how everything a run ships is named.

    Added rather than substituted, because a destination whose name already carries a dot (``model.v2``)
    would lose that half of itself to ``with_suffix``. One home, because every format and the record that
    describes them all land beside the same destination.
    """
    return destination.parent / f"{destination.name}.{suffix}"


def batch_may_vary(example: tuple[Tensor, ...]) -> tuple[tuple[Mapping[int, Any], ...]]:
    """How torch is told that the batch axis is free and every other size is settled.

    Declared rather than hoped for, which is the difference this route makes: a traced graph can bake in
    the batch it was traced at and say nothing, while an exported program carries the symbol, so an
    artifact written on two rows serves one. One symbol for every input, because they are one batch —
    two inputs of different lengths are not a batch this framework could have collated.

    Nested once for the graph's own signature: ``forward`` takes ``*tensors``, so torch sees a single
    parameter holding a tuple, and what is inside it is declared a level down.

    The symbol is the whole statement, and the range one could be given is not part of it: measured on
    torch 2.13, an artifact written under ``Dim("batch", min=2)`` serves one row just the same, so a
    minimum written here would read as a guarantee and hold nothing.
    """
    batch = torch.export.Dim("batch")
    return (tuple({BATCH_AXIS: batch} for _ in example),)


def _refuse_an_example_that_settles_the_batch(example: tuple[Tensor, ...]) -> None:
    """Every format writes its graph from an example, and every one generalizes the batch out of it.

    Measured on torch 2.13: an axis of size one is settled to exactly that size. ``torch.export`` answers
    "Constraints violated (batch)", which names neither the row nor the cure, and tracing answers nothing
    at all — it writes a file that serves one batch size for ever and looks perfect on the size it was
    written from.

    Here rather than in each backend: the reason is the same for all of them, it holds before any of
    their libraries is looked for, and the formats that cannot be measured on this machine need it most.
    """
    if any(tensor.shape[BATCH_AXIS] < WRITTEN_FROM for tensor in example):
        raise ValueError(
            "An artifact cannot be written from an example of one row: every format settles an axis of "
            "size one to exactly that size, so the file would serve one batch size for ever. Export from "
            f"at least {WRITTEN_FROM} rows."
        )


def _refuse_an_allowance_that_proves_nothing(atol: float, rtol: float) -> None:
    """A tolerance is a declaration like any other, and two of its values make verification a formality.

    Negative, and every share of the allowance is negative too: the worst value a comparison finds is
    still below zero, the reported figure stays at 0.0, and a file nobody has compared is described as
    standing exactly on the model. Zero on both, and an artifact that answers to the last bit of float
    is refused for using infinitely much of nothing. Infinite, and every artifact is inside it. Not a
    number, and it is below no bound at all, so the comparisons that would catch the other three are
    each false and it is accepted as though it were ordinary.

    Here rather than in the run that declares it, because both are the format's own contract and this is
    where every format's is: `tensorrt.yaml` is the shipped file that invites a run to raise them.
    """
    if not (math.isfinite(atol) and math.isfinite(rtol)) or atol < 0 or rtol < 0 or atol + rtol <= 0:
        raise ValueError(
            f"An allowance of atol {atol} and rtol {rtol} proves nothing: a negative one reports every "
            "artifact as exact whatever it answers, one of zero refuses an artifact that is exact, and "
            "an infinite one holds every artifact there is. Declare both finite and at or above zero, "
            "and at least one of them above it."
        )


class Exporter(ABC):
    """Writes a deployable graph to a file, and reads it back so that the file can be proven.

    ``load`` is abstract rather than optional: a format nobody can read back leaves behind an artifact
    that was never compared with the model it claims to be.

    Tolerances live here because they are knowledge of the format rather than of any one run: how much
    a written graph drifts follows from how it is written, and nothing a config could say changes it.
    """

    suffix: ClassVar[str]
    """The extension this format writes under; an artifact's name is built from it."""

    def __init__(self, atol: float = ABSOLUTE_TOLERANCE, rtol: float = RELATIVE_TOLERANCE) -> None:
        _refuse_an_allowance_that_proves_nothing(atol, rtol)
        self.atol = atol
        self.rtol = rtol

    def artifact_path(self, destination: Path) -> Path:
        """Where this format's artifact goes: ``destination`` under this format's own suffix."""
        return beside(destination, self.suffix)

    def travels_with(self, path: Path) -> tuple[Path, ...]:
        """Whatever else has to be beside ``path`` for it to be a model — nothing, for most formats.

        The artifact itself is not in the answer: whoever asks is holding it, and a list with the artifact
        first is a convention that has to be known to be read, so every reader would be free to forget it.
        A format that keeps its weights beside the graph says so here, because a deployment handed half of
        such a pair has a model that cannot answer and a file that looks whole.
        """
        return ()

    def describe(self, path: Path) -> Mapping[str, object]:
        """What a deployment needs to know about this artifact besides the files — read off the file.

        Nothing by default. A format with an operator set, a precision or a batch profile says it here,
        and says what was *written* rather than what a declaration asked for: the two can differ, and it
        is the artifact a deployment has to run.
        """
        return {}

    def export(self, graph: DeployableModel, example: tuple[Tensor, ...], destination: Path) -> Path:
        """Write ``graph`` at ``destination``, given without a suffix, and answer with the file written.

        Concrete on purpose: where an artifact goes, and who makes room for it, are answered once here
        rather than by each format — in the reference implementation each had its own copy of the naming
        rule and only one carried the reason. A backend states its ``suffix`` and writes what it is handed.
        """
        _refuse_an_example_that_settles_the_batch(example)
        path = self.artifact_path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.write(graph, example, path)
        return path

    def answers_at(self, written_at: int) -> tuple[int, ...]:
        """The batch sizes an artifact of this format has to answer at, given the one it was written from.

        The size it was written from, and one row: that is what a deployment sends, and it is the size an
        artifact that settled its batch axis fails at while answering the written one perfectly. A format
        whose artifact is built for a declared range of batches says so instead of this.
        """
        return (written_at, 1) if written_at != 1 else (1,)

    @abstractmethod
    def write(self, graph: DeployableModel, example: tuple[Tensor, ...], path: Path) -> None:
        """Put this format's bytes at ``path``, which its directory already exists for."""

    @abstractmethod
    def load(self, path: Path) -> Runnable:
        """Read back what ``write`` wrote, as something callable on the tensors the graph takes."""
