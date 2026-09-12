"""Stateless transformations remain functions; target geometry belongs to their adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from src.core import Batch, Geometry, Sample

if TYPE_CHECKING:
    from src.tasks import Task

type SampleTransform = Callable[[Sample], Sample]


@runtime_checkable
class GeometryAware(Protocol):
    """A transform that moves pixels asks which values move with them; the data build answers from the encoders."""

    def with_geometry(
        self,
        inputs: Mapping[str, Geometry],
        targets: Mapping[str, Geometry],
        auxiliary_inputs: Mapping[str, Geometry],
    ) -> SampleTransform: ...


@runtime_checkable
class AnswersTask(Protocol):
    """An augmentation whose draw is the supervision: it names the task it answers.

    Turning a picture makes the turn the answer, and cropping it makes "was this cropped" the answer.
    Such an augmentation has to say *which* task it answers, because the pipeline carrying it routes a
    value by its kind and so cannot tell two targets apart — and because a name is what lets the
    binding refuse a typo before the run rather than write nothing for the length of it.

    The task, not the column it reads: a sample carries one target per task, under the task's name,
    and the column is the data layer's business by the time a transform sees anything.
    """

    task: str


@runtime_checkable
class BatchTransform(Protocol):
    """Transforms one collated batch — the seam for an augmentation that mixes samples together.

    A ``SampleTransform`` cannot do this: while one sample is being loaded, the samples it would mix
    with do not exist yet. Mixing rewrites every task's target, so such a transform needs the run's
    tasks — and it is declared without them. It is bound when the run is set up, which is where a task
    it cannot serve is refused: before the first batch rather than an hour into one.

    Here rather than in ``core``, because it names ``Task``, which the core does not know: a contract
    sits with the lowest package that can spell it.
    """

    def for_tasks(self, tasks: Sequence[Task]) -> Callable[[Batch], Batch]:
        """The transform bound to these tasks; a new one, so what a declaration built stays as declared."""
        ...
