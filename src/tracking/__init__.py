"""Where a run's numbers and pictures go, and the one grammar they are named in.

Contracts, routing, and the backends themselves: importing this package is what makes their names
resolvable, and none of them pulls its client in until it is actually built.
"""

from __future__ import annotations

from src.tracking.base import DrawsMatrix, RecordsSummary
from src.tracking.clearml import ClearMLTracker
from src.tracking.keys import MetricKey, series
from src.tracking.report import report

__all__ = ["ClearMLTracker", "DrawsMatrix", "MetricKey", "RecordsSummary", "report", "series", "tracker_registry"]
