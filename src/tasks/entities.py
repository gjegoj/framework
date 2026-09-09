"""What a run learns: a named task of one kind, and the overrides a declaration may put on it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from src.core.entities import TaskFacts

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn

    from src.core.ports import Criterion
    from src.tasks.kinds import TaskKind


@dataclass(frozen=True, slots=True, eq=False)
class Task:
    """One learned objective: a kind, under a name, with the facts the data revealed about it.

    ``kind`` says what the task needs — encoder, head, loss, metrics, drawing; ``facts`` is
    what the data revealed about the target (class count, names, bin values), which sizes those
    parts. ``batch.targets[task.name]`` is the task's raw target. How predictions are
    produced is the model family's business.
    """

    name: str
    kind: TaskKind
    facts: TaskFacts = field(default_factory=TaskFacts)
    weight: float = 1.0
    lr: float | None = None
    """Own learning rate for this task's components — its head and its criterion.

    ``None`` shares the run's rate. Like ``weight``, a training knob is part of
    what a task *is*: how strongly it pulls, and how fast its own parts move.
    """

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Task name must be non-empty.")
        if self.weight <= 0:
            raise ValueError(f"Task weight must be positive, got {self.weight}.")
        if self.lr is not None and self.lr <= 0:
            raise ValueError(f"Task lr must be positive, got {self.lr}.")


type NativeHead = Literal["native"]
"""The reserved head name: not a class in ``head_registry`` but the head the backbone brings.

Interpreted by the composition root before any registry lookup, so ``head: {name: native}``
and its sugar ``head: native`` are one declaration; ``native`` takes no arguments. The type is
the one place the word is spelled and ``NATIVE_HEAD`` is checked against it — measured: mypy 2.3
rejects ``Literal[NATIVE_HEAD]`` over a ``Final``, so the alias is the declaration, not the constant.
"""

NATIVE_HEAD: NativeHead = "native"


@dataclass(frozen=True, slots=True)
class Overrides:
    """What a task declaration may put in place of its kind's defaults.

    The three keys of the task section that override a component: ``head`` chooses the
    head — a factory for a kind to build, or ``NATIVE_HEAD`` for the one the backbone
    brings — ``streams`` what it reads, ``loss`` the criterion. Factories rather than
    built objects, because the sizes they need are resolved by the kind from the backbone
    and the facts. Absent means the kind's own answer.
    """

    head: Callable[[int | tuple[int, ...], int], nn.Module] | NativeHead | None = None
    streams: tuple[str, ...] | None = None
    loss: Callable[[TaskFacts, int], Criterion] | None = None
    """Given the facts and the width of the first stream read (an embedding size)."""
