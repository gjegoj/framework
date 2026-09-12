"""A `tracker` declaration becomes the place a run is recorded — or nothing at all."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from lightning.pytorch.loggers import CSVLogger, Logger

from src.config import ComponentConfig
from src.config.instantiate import resolve_factory
from src.tracking import ClearMLTracker
from src.tracking.build import build_tracker
from src.tracking.registry import tracker_registry


def test_a_run_that_declares_no_tracker_records_nowhere() -> None:
    """`tracker: none` is a declaration, and the run says so by having nothing to log to."""
    assert build_tracker(None) is None


def test_the_name_a_declaration_writes_becomes_the_tracker_it_names(tmp_path: Path) -> None:
    built = build_tracker(ComponentConfig.model_validate({"name": "csv", "save_dir": str(tmp_path)}))

    assert isinstance(built, CSVLogger) and built.save_dir == str(tmp_path)


@pytest.mark.parametrize("name", sorted(tracker_registry))
def test_every_registered_name_is_something_lightning_can_log_to(name: str) -> None:
    resolved = resolve_factory(ComponentConfig(name=name), tracker_registry)

    assert isinstance(resolved, type) and issubclass(resolved, Logger)


def test_something_that_is_not_a_tracker_is_refused_where_it_was_declared() -> None:
    with pytest.raises(TypeError, match="Counter"):
        build_tracker(ComponentConfig.model_validate({"_target_": "collections.Counter"}))


def test_a_backend_is_named_long_before_its_client_is_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing this package registers every tracker; a run that declares none installs none.

    Nor does declaring one, until it is used: what needs the client is starting the run on the service.
    """
    monkeypatch.setitem(sys.modules, "clearml", None)

    resolved = resolve_factory(ComponentConfig(name="clearml"), tracker_registry)

    assert isinstance(resolved, type) and issubclass(resolved, ClearMLTracker)
    with pytest.raises(ImportError):
        _ = resolved().experiment
