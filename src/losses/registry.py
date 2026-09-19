"""The names a loss declaration may write."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core import Registry

if TYPE_CHECKING:
    from src.losses.base import Loss

loss_registry: Registry[Loss] = Registry("loss")
"""What `tasks.<name>.loss` writes; a loss of your own is reached by ``_target_``, or named here
as ``Registry`` describes — a decorator alone leaves the name unresolvable until the module runs."""

distillation_loss_registry: Registry[Loss] = Registry("distillation loss")
"""What `learner.loss` writes: how far a student's answer is from the teacher's.

Its own names rather than the ones above, because a registry belongs to a position and these are two.
What a task is judged by reads a target the data settled; this reads a second network's answer to the
same batch. A name answering one of those questions answers nothing about the other, and sharing the
list would let either be declared where it means nothing — with no refusal, since both are losses and
both would build.
"""
