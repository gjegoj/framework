"""The data side of an experiment, as the training package sees it: per-stage datasets plus facts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from src.data.statistics import DatasetStatistics

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch.utils.data import Dataset

    from src.core.entities import DatasetFacts, Sample
    from src.core.taxonomy import Stage


class DataModule(ABC):
    """The data side of an experiment: per-stage datasets plus inferred facts.

    The one data port in the core, because it is the data ↔ training boundary; the ports
    that would drag an I/O library in (sources, encoders, trackers) stay with their packages.
    """

    @abstractmethod
    def setup(self) -> DatasetFacts:
        """Prepare the per-stage datasets and return what the data revealed about each task.

        Runs before tasks and heads are built — the ordering that lets output sizes come
        from data instead of config — and says so by its return value.
        """

    @abstractmethod
    def dataset(self, stage: Stage) -> Dataset[Sample]:
        """Return the dataset for ``stage``; ``setup`` must have run first.

        Raises ``LookupError`` naming the stages it does have when it has none for this one —
        an answer, not a failure: a pipeline may legitimately carry no test data, and the
        consumer says what it does instead.
        """

    def statistics(self) -> DatasetStatistics:
        """What this pipeline is about to serve, for the report drawn before epoch one.

        Concrete with an empty default: a pipeline that cannot describe its data answers with
        nothing, and the report still names it.
        """
        return DatasetStatistics()


def require_stage[T](datasets: Mapping[Stage, T] | None, stage: Stage, owner: str) -> T:
    """One stage's dataset, or the two refusals :meth:`DataModule.dataset` documents.

    A free function rather than a template method, so a lazy or streaming pipeline that holds
    no dict of stages is not forced to have one.

    Parameters:
        datasets (Mapping[Stage, T] | None): What ``setup`` built, or ``None`` before it ran.
        stage (Stage): The stage being asked for.
        owner (str): The pipeline's own name, for the message.
    """
    if datasets is None:
        raise RuntimeError(f"{owner}.setup() must run before requesting datasets.")
    try:
        return datasets[stage]
    except KeyError:
        available = ", ".join(datasets) or "none"
        raise LookupError(f"No dataset for stage '{stage}'. Available stages: {available}.") from None
