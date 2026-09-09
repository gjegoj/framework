"""The target encoder contract: ``load`` before the transforms, ``encode`` after."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Self, override

from src.core.entities import TaskFacts
from src.core.taxonomy import Geometry
from src.data.statistics import Distribution

if TYPE_CHECKING:
    from src.data.cache import LoaderCache

log = logging.getLogger(__name__)


class TargetEncoder(ABC):
    """Turns one task's target column into training data, in two halves.

    ``load`` runs *before* the sample transforms (one table cell in, the form the transforms
    should see out; identity by default, a file read for a mask). ``encode`` runs *after*
    them, on whatever value survived, so an augmentation may write a raw class name or a
    number and this encoder makes training sense of it. ``fit`` learns vocabulary or
    statistics from the training split; what it inferred (``num_classes``, ``class_names``)
    is what ``setup`` returns as the task's facts. ``validate`` runs over every other split, so
    a value the fitted encoder cannot encode is refused at setup rather than by the epoch that
    meets it. ``geometry`` says how a value is transformed with the image; ``NONE`` values never
    enter the pipeline.
    """

    geometry: ClassVar[Geometry] = Geometry.NONE

    def fit(self, values: Iterable[Any]) -> Self:
        """Learn from training-split values and hand the fitted encoder back. Default: nothing to learn."""
        return self

    def validate(self, values: Iterable[Any]) -> None:
        """Refuse what ``encode`` could not serve, over a split ``fit`` never saw. Default: nothing to refuse."""
        return

    def load(self, value: Any) -> Any:
        """One table cell into the form the transforms see. Default: as it stands."""
        return value

    @abstractmethod
    def encode(self, value: Any) -> Any:
        """The post-transform value into the target's training form."""

    @property
    def num_classes(self) -> int | None:
        """Label-vocabulary size, ``None`` for class-free targets."""
        return None

    @property
    def class_names(self) -> list[str] | None:
        """Class names aligned with encoded indices, ``None`` when class-free."""
        return None

    @property
    def class_values(self) -> list[float] | None:
        """The number each encoded position stands for, ``None`` when unordered.

        Set by encoders that spread one continuous value over ordered classes:
        the values are what turns a predicted distribution back into a number.
        """
        return None

    def distribution(self, values: Iterable[Any]) -> Distribution | None:
        """What this column holds, or ``None`` when this encoder does not describe it.

        Beside ``facts()``, and for the same reason: the encoder owns the
        vocabulary and the parsing, so nothing else can count its own column
        correctly. A method on the base class rather than an optional capability
        bolted on — an encoder that says nothing returns ``None`` and the report
        names the task anyway, where a missing method dropped the column in
        silence and left the reader to guess which of their targets was gone.
        """
        return None

    def facts(self) -> TaskFacts:
        """What fitting this encoder inferred, as the one value ``setup`` returns for its task.

        Reporting the facts together is what keeps a caller from enumerating
        them: a new kind of fact is then declared by the encoders that have it,
        not by everything that reports facts.
        """
        return TaskFacts(
            num_classes=self.num_classes,
            class_names=tuple(names) if (names := self.class_names) is not None else None,
            class_values=tuple(values) if (values := self.class_values) is not None else None,
        )


class VocabularyTargetEncoder(TargetEncoder):
    """An encoder whose target is read against a declared class vocabulary.

    The base says so, and that is what ``build_target_encoder`` reads: ``classes`` is handed
    to these and refused on every other encoder (ADR-0004).
    The vocabulary is the contract the data is validated against, and the index space the
    model's outputs live in.

    Parameters:
        classes (Mapping[int, str]): The vocabulary, index to name.
    """

    def __init__(self, classes: Mapping[int, str]) -> None:
        self._names = _ordered_names(classes)
        self._positions = {name: position for position, name in enumerate(self._names)}

    @override
    def fit(self, values: Iterable[Any]) -> Self:
        """A declared vocabulary learns nothing: fitting is validating the train split against it."""
        self.validate(values)
        return self

    @property
    def num_classes(self) -> int:
        return len(self._names)

    @property
    def class_names(self) -> list[str]:
        return list(self._names)


class FileTargetEncoder(TargetEncoder):
    """An encoder that reads a file behind a loader of its own.

    The base says so, and that is what ``build_target_encoder`` reads: a cache reaches these through
    ``use_cache`` after construction, and no other encoder, because nothing else has a
    read to serve from memory.
    """

    @abstractmethod
    def use_cache(self, cache: LoaderCache) -> None:
        """Serve this encoder's reads from ``cache`` from now on."""


def _ordered_names(classes: Mapping[int, str]) -> list[str]:
    """Index-keyed names as the list their order means; loud on a broken range.

    A declared vocabulary is the index space the model's outputs live in: a gap shifts every
    class above it, a repeated name makes two classes indistinguishable in every report, and
    both fail silently. Checked where the vocabulary is consumed — the encoder's contract, refused
    when the encoder is built, before a row is encoded.

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
