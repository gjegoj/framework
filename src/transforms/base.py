"""Stateless transformations remain functions; target geometry belongs to their adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from src.core import Batch, Geometry, Normalization, Sample

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

    Turning an image makes the turn the answer, and cropping it makes "was this cropped" the answer.
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


@runtime_checkable
class AppliesNormalization(Protocol):
    """A transform that can say what fixed scaling it leaves an image's values in.

    Two declarations own the two halves of one fact — an encoder says what a model was trained under,
    a stage's chain says what the pixels actually go through — and nothing but a run assembling both can
    see them disagree. This is how the second half answers.

    Optional, as a capability with no sensible default is: a transform that cannot say is not asked, and
    one reached by ``_target_`` is the run's own business.
    """

    @property
    def normalization(self) -> Normalization | None: ...


def is_the_same_scaling(declared: Normalization, applied: Normalization | None) -> bool:
    """Whether a chain applies what an input declares, reading a single number the way the library does.

    ``albumentations.Normalize(mean=0.5)`` spreads one number over every channel, and this package keeps
    it as the single number it was given — see ``_per_channel`` — because only here is it known that this
    is what the library means by it. Whoever compares the two halves of the scaling therefore asks this
    rather than restating the rule: a run that declares the same number per channel is a run that is
    right, and refusing it would be refusing the truth.
    """
    if applied is None:
        return False
    return _spread(declared.mean, applied.mean) and _spread(declared.std, applied.std)


def _spread(declared: tuple[float, ...], applied: tuple[float, ...]) -> bool:
    return applied == declared or (len(applied) == 1 and set(declared) == set(applied))
