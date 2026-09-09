"""One declared criterion, or several added with their weights."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config.instantiate import resolve_params, resolve_target
from src.config.tasks import LossConfig
from src.core.ports import Criterion
from src.losses.composite import WeightedSumCriterion
from src.losses.registry import criterion_registry

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.core.entities import TaskFacts


def build_criterion(
    declared: LossConfig | Sequence[LossConfig], facts: TaskFacts | None = None, embedding_dim: int | None = None
) -> Criterion:
    """The declared criterion — one, or the weighted sum of several.

    The weight sits on the declaration, so a term keeps it wherever it is used; a single
    unweighted part is itself. A criterion sized by the task's facts says so on its class
    (``sized``) and is built from them — an expectation term gets its bin centres from the
    encoder that laid them out, a proxy criterion sizes its prototypes from the vocabulary
    and the stream the task reads — and nobody pastes either into config a second time.
    Every other criterion is built from its declaration alone.
    """
    parts = [declared] if isinstance(declared, LossConfig) else list(declared)
    built: list[tuple[Criterion, float]] = [(_criterion(part, facts, embedding_dim), part.weight) for part in parts]
    if len(built) == 1 and built[0][1] == 1.0:
        return built[0][0]
    return WeightedSumCriterion(built)


def _criterion(declared: LossConfig, facts: TaskFacts | None, embedding_dim: int | None) -> Criterion:
    """One part: through ``sized`` where the class declares it, the plain constructor otherwise."""
    factory = resolve_target(declared, criterion_registry)
    params = resolve_params(declared)
    sized = getattr(factory, "sized", None)
    if sized is None:
        built = factory(**params)
        if not isinstance(built, Criterion):
            # Measured before this check: a raw torch loss by `_target_` built, and died in the first
            # training step when its bare tensor met a reader expecting a Loss.
            raise TypeError(
                f"'{declared.spelled}' built {type(built).__name__}, which is not a Criterion: a criterion returns a "
                f"Loss with a named part, and a torch loss returns a bare tensor. Wrap it in a WrappedCriterion "
                f"subclass with a part_name (see src/losses/regression.py), or name a registered criterion."
            )
        return built
    if facts is None or embedding_dim is None:
        raise ValueError(
            f"'{declared.spelled}' is sized from a task's facts, and this position has none: "
            f"a distillation loss compares logits as they are."
        )
    built_sized: Criterion = sized(facts, embedding_dim, **params)
    return built_sized
