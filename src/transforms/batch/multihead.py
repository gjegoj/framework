"""Multi-head adapters over torchvision ``v2.MixUp`` / ``v2.CutMix``.

The torchvision transforms mix one image with one label tensor sized by a single
``num_classes``. A multi-task batch has one shared image but several heads, each
with its own class count. These subclasses reuse torchvision's whole machinery —
the Beta sampling, the CutMix box geometry, and the image mix/paste — and add a
single thing: mixing *several* heads' labels with the **same** sampled params, by
one-hot-encoding each head to its own ``num_classes`` first (so torchvision's
label mix no longer needs a single global ``num_classes``).
"""

from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.nn.functional import one_hot
from torchvision.transforms import v2

# Sentinel placed in params["labels"] so the image is never routed to the label branch.
_NOT_A_LABEL = object()


class _MultiHeadMix:
    """Mixin (combined with a ``v2`` MixUp/CutMix) that mixes one image + many labels."""

    def mix(
        self,
        image: Tensor,
        labels: dict[str, Tensor],
        num_classes: dict[str, int],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return ``(mixed_image, {key: soft_label})`` sharing one sampled mix."""
        params: dict[str, Any] = self.make_params([image])  # type: ignore[attr-defined]
        params["batch_size"] = image.shape[0]
        params["labels"] = _NOT_A_LABEL
        mixed_image: Tensor = self.transform(image, params)  # type: ignore[attr-defined]

        lam = self._label_lam(params)
        mixed = {
            key: self._mixup_label(self._as_soft(label, num_classes[key]), lam=lam)  # type: ignore[attr-defined]
            for key, label in labels.items()
        }
        return mixed_image, mixed

    def _label_lam(self, params: dict[str, Any]) -> float:
        """The mixing weight applied to labels (differs for MixUp vs CutMix)."""
        raise NotImplementedError

    @staticmethod
    def _as_soft(label: Tensor, num_classes: int) -> Tensor:
        """One-hot a class-index label to ``[B, C]`` float (pass-through if already 2-D)."""
        if label.ndim == 1:
            return one_hot(label.long(), num_classes).float()
        return label.float()


class MixUpMultiHead(_MultiHeadMix, v2.MixUp):
    """MixUp that mixes one image and any number of heads' labels with shared ``lam``."""

    def __init__(self, alpha: float = 0.2) -> None:
        super().__init__(alpha=alpha, num_classes=None)

    def _label_lam(self, params: dict[str, Any]) -> float:
        return float(params["lam"])


class CutMixMultiHead(_MultiHeadMix, v2.CutMix):
    """CutMix that pastes one patch and mixes any number of heads' labels by area."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__(alpha=alpha, num_classes=None)

    def _label_lam(self, params: dict[str, Any]) -> float:
        return float(params["lam_adjusted"])
