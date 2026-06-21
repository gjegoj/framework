"""Shared bases for custom Albumentations augmentations.

These plug into an ``albumentations.Compose`` via ``_target_`` and are framework-agnostic
(only albumentations + numpy). ``LabelAwareDualTransform`` is the universal base for any
geometric augmentation that must also rewrite a discrete label — subclass it from any
augmentation module, not just rotation.
"""

from __future__ import annotations

from typing import Any

import albumentations as A


class LabelAwareDualTransform(A.DualTransform):
    """Base for geometric augmentations that also rewrite a discrete label.

    The label is bound to a *configurable* data key (``label_key``) instead of one fixed by a
    method name, so the same transform composes with any task — the config points it at the
    task's ``target`` column. Subclasses implement ``apply`` (image), optionally ``apply_to_mask``,
    and ``apply_to_label`` (the label rule); all three receive the same sampled ``get_params``,
    so the image and its label always agree.

    Parameters:
        label_key (str): Data key of the label to rewrite — the task's ``target`` column.
        p (float): Probability of applying the transform.
    """

    def __init__(self, label_key: str, p: float = 1.0) -> None:
        self._label_key = label_key  # set before super().__init__ so _set_keys can register it
        super().__init__(p=p)

    def _set_keys(self) -> None:
        super()._set_keys()  # registers the built-in image / mask keys
        self._available_keys.add(self._label_key)
        self._key2func[self._label_key] = self.apply_to_label

    def apply_to_label(self, label: Any, **params: Any) -> Any:
        """Rewrite the label using the same sampled params as ``apply`` (subclasses override)."""
        raise NotImplementedError
