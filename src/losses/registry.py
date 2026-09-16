"""The names a loss declaration may write."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core import Registry

if TYPE_CHECKING:
    from src.losses.base import Loss

loss_registry: Registry[Loss] = Registry("loss")
"""What `tasks.<name>.loss` writes; a loss of your own is reached by ``_target_``, or named here
as ``Registry`` describes — a decorator alone leaves the name unresolvable until the module runs."""
