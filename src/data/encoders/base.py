"""The target encoder contract: ``load`` before the transforms, ``encode`` after."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, ClassVar

from src.core.entities import Distribution, TargetFacts
from src.core.taxonomy import Geometry

log = logging.getLogger(__name__)


class TargetEncoder(ABC):
    """Turns one task's target column into training data, in two halves.

    ``load`` runs *before* the sample transforms (one table cell in, the form the transforms
    should see out; identity by default, a file read for a mask). ``encode`` runs *after*
    them, on whatever value survived, so an augmentation may write a raw class name or a
    number and this encoder makes training sense of it. ``fit`` learns vocabulary or
    statistics from the training split; what it inferred (``num_classes``, ``class_names``)
    is recorded into the ``DataProfile``. ``geometry`` says how a value is transformed with
    the image; ``NONE`` values never enter the pipeline.
    """

    geometry: ClassVar[Geometry] = Geometry.NONE

    def fit(self, values: Iterable[Any]) -> None:
        """Learn from training-split values. Default: nothing to learn."""

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

    def facts(self) -> TargetFacts:
        """What fitting this encoder inferred, as the one record a profile stores.

        Reporting the facts together is what keeps a caller from enumerating
        them: a new kind of fact is then declared by the encoders that have it,
        not by everything that fills a ``DataProfile``.
        """
        return TargetFacts(
            num_classes=self.num_classes,
            class_names=self.class_names,
            class_values=self.class_values,
        )
