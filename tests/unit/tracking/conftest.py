"""A stand-in for ClearML: the adapter is under test, not the service behind it."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Any

import pytest


@dataclass
class Recorded:
    """Everything the stubbed backend was asked to record, in the order it was asked."""

    scalars: list[tuple[str, str, float, int]] = field(default_factory=list)
    matrices: list[dict[str, Any]] = field(default_factory=list)
    singles: dict[str, float] = field(default_factory=dict)
    media: list[dict[str, Any]] = field(default_factory=list)
    histograms: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    started: dict[str, Any] = field(default_factory=dict)
    destinations: list[str] = field(default_factory=list)
    connected: dict[str, Any] = field(default_factory=dict)
    flushes: int = 0
    fails_to_flush: bool = False
    waited: list[str] = field(default_factory=list)
    fails_to_upload: str | None = None
    declines_upload: str | None = None


@pytest.fixture
def clearml(monkeypatch: pytest.MonkeyPatch) -> Recorded:
    """A ``clearml`` module in ``sys.modules`` whose task records what a logger reports to it.

    It carries a real ``__spec__`` on purpose: other libraries probe for clearml with
    ``importlib.util.find_spec``, which raises rather than answering "no" when a module sitting in
    ``sys.modules`` has none.
    """
    recorded = Recorded()

    class Backend:
        def report_scalar(self, title: str, series: str, value: float, iteration: int) -> None:
            recorded.scalars.append((title, series, value, iteration))

        def report_confusion_matrix(self, **reported: Any) -> None:
            recorded.matrices.append(reported)

        def report_single_value(self, name: str, value: float) -> None:
            recorded.singles[name] = value

        def report_media(self, **reported: Any) -> None:
            recorded.media.append({**reported, "read": reported["stream"].read()})

        def report_histogram(self, **reported: Any) -> None:
            recorded.histograms.append(reported)

        def set_default_upload_destination(self, uri: str) -> None:
            recorded.destinations.append(uri)

    class Task:
        name = "a-run"
        id = "abc123"

        @classmethod
        def init(cls, **options: Any) -> Task:
            recorded.started = options
            return cls()

        def get_logger(self) -> Backend:
            return Backend()

        def upload_artifact(self, name: str, artifact_object: Any, wait_on_upload: bool = False) -> bool:
            if name == recorded.fails_to_upload:
                raise RuntimeError("the service is unreachable")
            if name == recorded.declines_upload:
                return False
            recorded.artifacts[name] = artifact_object
            if wait_on_upload:
                recorded.waited.append(name)
            return True

        def connect(self, values: dict[str, Any]) -> None:
            recorded.connected = values

        def flush(self) -> None:
            if recorded.fails_to_flush:
                raise RuntimeError("the service is unreachable")
            recorded.flushes += 1

    module = ModuleType("clearml")
    module.__spec__ = ModuleSpec("clearml", loader=None)
    module.Task = Task  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "clearml", module)
    return recorded
