"""Shared bases for custom Albumentations augmentations.

These plug into an ``albumentations.Compose`` via ``_target_`` and are framework-agnostic
(only albumentations + numpy). ``LabelAwareMixin`` binds a *configurable* discrete-label target
to an ``apply_to_label`` rule on top of Albumentations' own ``CustomTransformsApplyMixin`` (which
auto-registers ``apply_to_<key>`` methods). Mix it into **any** albumentations transform — one
built from scratch (``LabelAwareDualTransform`` below) or an existing library transform — to make
that transform also rewrite a label; place it *before* the transform base in the inheritance list.
"""

from __future__ import annotations

from typing import Any

import albumentations as A


class LabelAwareMixin(A.CustomTransformsApplyMixin):
    """Register a configurable discrete label as a transform target routed to ``apply_to_label``.

    Albumentations' ``CustomTransformsApplyMixin`` already discovers an ``apply_to_label`` method
    and registers it under the fixed data key ``"label"``. This mixin adds the one thing the
    framework needs: the label is bound to a *configurable* key (``label_key``) so the same
    augmentation composes with any task — the config points it at the task's ``target`` column
    instead of a name baked into the method.

    It carries no ``__init__`` (only ``_set_keys``), so it composes with any albumentations
    transform's constructor. Set ``self.label_key`` *before* calling ``super().__init__(...)`` and
    the registration picks it up; subclasses implement ``apply_to_label``.
    """

    label_key: str = "label"  # subclasses set self.label_key before super().__init__()

    def _set_keys(self) -> None:
        super()._set_keys()  # auto-registers apply_to_label under the default "label" key
        if self.label_key != "label":
            self._key2func[self.label_key] = self._key2func.pop("label")
            self._available_keys.discard("label")
            self._available_keys.add(self.label_key)

    def apply_to_label(self, label: Any, **params: Any) -> Any:
        """Rewrite the label using the same sampled params as ``apply`` (subclasses override)."""
        raise NotImplementedError


class LabelAwareDualTransform(LabelAwareMixin, A.DualTransform):
    """Base for geometric augmentations built from scratch that also rewrite a discrete label.

    A convenience pairing of ``LabelAwareMixin`` with ``DualTransform`` for the common case of an
    augmentation that supplies its own ``apply`` (image), optionally ``apply_to_mask``, and
    ``apply_to_label`` (the label rule); all three receive the same sampled ``get_params``, so the
    image and its label always agree. To label-augment an *existing* library transform instead,
    mix ``LabelAwareMixin`` directly into it.

    Parameters:
        label_key (str): Data key of the label to rewrite — the task's ``target`` column.
        p (float): Probability of applying the transform.
    """

    def __init__(self, label_key: str, p: float = 1.0) -> None:
        self.label_key = label_key  # set before super().__init__ so _set_keys can register it
        super().__init__(p=p)
