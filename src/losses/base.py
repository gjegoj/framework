"""Shared base for single-part criteria — the extension point for wrapping any loss."""

from __future__ import annotations

from typing import ClassVar

from torch import Tensor, nn

from src.core.entities import Loss
from src.core.ports import Criterion


def split_views(logits: Tensor, count: int, owner: str) -> tuple[Tensor, ...]:
    """Validate a stacked ``[B, count, D]`` carrier and return its views.

    Shared by the families that consume view carriers (contrastive, ranking),
    so a shape mistake names the criterion instead of surfacing deep inside a
    torch loss.
    """
    if logits.dim() != 3 or logits.size(1) != count:
        raise ValueError(f"{owner} expects stacked embeddings [B, {count}, D], got {tuple(logits.shape)}.")
    return tuple(logits.unbind(dim=1))


class WrappedCriterion(Criterion):
    """Base for criteria that wrap one torch loss and log it as one named part.

    **Wrap a module, subclass a composer.** Math that ends in one tensor-in →
    tensor-out ``nn.Module`` — torch's, smp's, or written beside the wrapper — belongs
    here; a criterion that composes *other criteria*, like a distance slot or a weighted
    sum, subclasses ``Criterion`` directly, because its children already return ``Loss``.
    Either way the logged part is ``part_name``, never an inline string.

    Subclasses build their loss module in ``__init__`` and hand it over. The wrapped
    module is a registered submodule, so its parameters train and its buffers move
    across devices. A new loss goes into its family module (``classification``,
    ``regression``, ``segmentation``, ``contrastive``, ...) and registers in the
    ``criterion_registry`` under a config-facing name.

    Declare explicitly only the parameters that need conversion — a YAML list into a
    tensor — or a framework default; forward everything else verbatim through
    ``**kwargs``, so every upstream knob stays reachable without a wrapper change.
    """

    part_name: ClassVar[str]

    def __init__(self, loss: nn.Module) -> None:
        super().__init__()
        self._loss = loss

    def forward(self, logits: Tensor, target: Tensor) -> Loss:
        logits, target = self._prepare(logits, target)
        value: Tensor = self._loss(logits, target)
        return Loss.part(self.part_name, value)

    def _prepare(self, logits: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        """Shape/type hook applied before the wrapped loss; identity by default."""
        return logits, target
