"""The names a loss declaration may write."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core import Registry

if TYPE_CHECKING:
    from src.losses.base import Loss

loss_registry: Registry[Loss] = Registry("loss")
"""What `tasks.<name>.loss` writes; register with ``@loss_registry.register("name")``."""
