"""Registry of loss criteria, selectable from YAML by ``name`` (or a ``_target_`` spec).

Lives in its own module (mirroring ``models``/``metrics``/``export``) so any criterion file
— and external code — can register against it without importing the standard criteria.
"""

from __future__ import annotations

from src.core.ports import Criterion
from src.core.registry import Registry

criteria: Registry[Criterion] = Registry("criterion")
