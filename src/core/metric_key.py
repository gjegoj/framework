"""MetricKey: the single parser of the logged-scalar key grammar.

Every scalar logged during training is keyed by a small grammar built from the
tokens in :mod:`src.core.keys`:

- task metrics — ``{task}/{metric}/{stage}`` (scalar) or
  ``{task}/{metric}/{stage}/{leaf}`` where ``leaf`` is ``mean`` (a vector
  metric's average) or a class name (a per-class value);
- losses — ``loss/{stage}/{component}`` where ``component`` is ``total`` (the
  aggregate) or ``{task}/{loss_name}`` (a per-task term, so it may contain ``/``).

This grammar used to be re-parsed independently by the metrics summary, the
progress bar, and the ClearML logger — one design decision leaked across three
modules. ``MetricKey`` owns it: callers parse once and read named fields /
projections instead of splitting ``/`` and special-casing ``loss`` themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.enums import Stage
from src.core.keys import LOSS

# A stage token recognised inside a key. ``Stage`` is a ``StrEnum``, so a plain
# string segment compares/hashes equal to its member ("val" in this set is True).
_STAGES: frozenset[Stage] = frozenset(Stage)


@dataclass(frozen=True, slots=True)
class MetricKey:
    """A parsed logged-scalar key, classified as a loss or a task metric.

    Parameters:
        raw (str): The original key, unchanged.
        is_loss (bool): True for the ``loss/{stage}/{component}`` form.
        task (str | None): The task name for a metric key; ``None`` for a loss
            key or an unclassified key.
        name (str): The metric name (metric form) or the loss component (loss
            form). For an unclassified key this is ``raw``.
        stage (Stage | None): The recognised stage, or ``None`` when the key has
            no stage segment (e.g. a 2-part key, or an unknown 3rd segment).
        leaf (str | None): A vector metric's 4th segment — ``mean`` or a class
            name — for the metric form; ``None`` otherwise.
    """

    raw: str
    is_loss: bool
    task: str | None
    name: str
    stage: Stage | None
    leaf: str | None

    @classmethod
    def parse(cls, key: str) -> MetricKey:
        """Parse a logged-scalar key into its classified components.

        Parameters:
            key (str): A logged scalar key (e.g. ``loss/val/total``, ``species/f1/val``).

        Returns:
            MetricKey: The classified key. Keys matching neither form are returned
            unclassified (``stage`` is ``None``, ``name`` is ``raw``).
        """
        parts = key.split("/")
        if parts[0] == LOSS and len(parts) >= 3 and parts[1] in _STAGES:
            return cls(raw=key, is_loss=True, task=None, name="/".join(parts[2:]), stage=Stage(parts[1]), leaf=None)
        if len(parts) in (3, 4) and parts[2] in _STAGES:
            return cls(
                raw=key,
                is_loss=False,
                task=parts[0],
                name=parts[1],
                stage=Stage(parts[2]),
                leaf=parts[3] if len(parts) == 4 else None,
            )
        return cls(raw=key, is_loss=False, task=None, name=key, stage=None, leaf=None)

    @property
    def display_name(self) -> str:
        """The key with both the stage and any vector leaf stripped.

        ``loss/val/total`` → ``loss/total``; ``species/f1/val`` → ``species/f1``;
        ``breed/f1/test/mean`` → ``breed/f1``. An unclassified key returns ``raw``.
        Used for end-of-run summary rows and progress-bar row identity.
        """
        if self.is_loss:
            return f"{LOSS}/{self.name}"
        if self.task is None:
            return self.raw
        return f"{self.task}/{self.name}"

    def without_stage(self) -> str:
        """The key with only the stage segment removed (any leaf kept).

        ``label/accuracy/train`` → ``label/accuracy``; ``loss/val/total`` →
        ``loss/total``; ``breed/f1/test/Abyssinian`` → ``breed/f1/Abyssinian``.
        A key with no stage is returned unchanged.
        """
        if self.is_loss:
            return f"{LOSS}/{self.name}"
        if self.stage is None:
            return self.raw
        base = f"{self.task}/{self.name}"
        return f"{base}/{self.leaf}" if self.leaf is not None else base

    def without_leaf(self) -> str:
        """The key with only the trailing vector leaf removed (the stage kept).

        ``breed/f1/test/mean`` → ``breed/f1/test``; a key without a leaf
        (``seg/iou/val``) is returned unchanged. Used for the progress-bar row
        identity, where a vector metric's mean shares a row with its stage.
        """
        if self.leaf is None:
            return self.raw
        return self.raw.rsplit("/", 1)[0]
