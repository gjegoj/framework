"""Target encoders (data-layer I/O): turn a raw target value into a model-ready tensor.

Two steps, uniform across every encoder, straddling the per-sample transform:
  1. ``load(value)``    — pre-transform: return the *encoded, transform-ready*
                          representation (label → class index, multi-label → multi-hot
                          vector, mask → array). Because it rides through the transform
                          pipeline, the same augmentation that resizes/flips a mask can
                          also update a label — e.g. a rotation aug that bumps the
                          rotation class via its ``apply_to_<task>`` hook.
  2. ``to_tensor(val)`` — post-transform: tensorize the (possibly transform-modified)
                          value into a final model-ready tensor (fixing dtype).

Keeping both steps uniform means ``Dataset.__getitem__`` has a single clean loop for
each stage instead of branching on encoder type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor

from src.data.registry import target_encoders
from src.data.statistics import CategoricalDistribution, ContinuousDistribution, Distribution, Histogram

# Number of bins for the regression value histogram (capped by the sample count).
_REGRESSION_HISTOGRAM_BINS = 20


class TargetEncoder(ABC):
    """Two-step encoder: ``load`` (pre-transform, encodes) → ``to_tensor`` (post-transform, tensorizes).

    Two orthogonal flags describe what ``load`` consumes / produces:

    - ``file_based`` — the column value is a file path: the caller prepends ``root_path``
      before ``load``, records it as the target's source, and caches the file read. Mirrors
      ``InputLoader.file_based`` on the input side.
    - ``spatial`` — ``load`` returns a numpy array (a mask) that must ride through the same
      geometric transform as the image (registered as an Albumentations ``mask`` target).

    They coincide for ``MaskEncoder`` (a mask is both a file and spatial) but are independent:
    a target read from a file yet not transformed geometrically would be ``file_based`` only.
    """

    file_based: bool = False
    spatial: bool = False

    @abstractmethod
    def fit(self, values: Iterable[Any]) -> None:
        """Learn any state needed to encode (e.g. the class vocabulary)."""

    @abstractmethod
    def load(self, value: Any) -> Any:
        """Pre-transform step: return the encoded, transform-ready representation.

        Labels → the class index; multi-label → a multi-hot vector; masks (spatial) →
        the array read from disk. The result rides through the transform pipeline, so the
        same augmentation that resizes/flips a mask can also update a label.
        """

    @abstractmethod
    def to_tensor(self, value: Any) -> Tensor:
        """Post-transform step: tensorize the (possibly transform-modified) value into a model-ready tensor."""

    @property
    def num_classes(self) -> int | None:
        """Number of classes if categorical, else ``None``."""
        return None


class _CategoricalEncoder(TargetEncoder):
    """Shared vocabulary handling for label encoders: the index⇄label maps + validation.

    Subclasses supply the encoding (``load`` → class index / multi-hot) and pick which
    labels each row contributes to ``fit`` (a single label, or many for multi-label).

    Parameters:
        class_mapping (dict[int, str] | None): Fixed index→label map; if provided ``fit`` only
            validates, and the vocabulary is not inferred from data.
    """

    def __init__(self, class_mapping: dict[int, str] | None = None) -> None:
        self._index_to_label: dict[int, str] = {}
        self._label_to_index: dict[str, int] = {}
        if class_mapping is not None:
            self._set_mapping([class_mapping[i] for i in sorted(class_mapping)])

    def _set_mapping(self, labels: list[str]) -> None:
        self._index_to_label = dict(enumerate(labels))
        self._label_to_index = {label: index for index, label in self._index_to_label.items()}

    def _require_mapping(self) -> None:
        if not self._label_to_index:
            raise ValueError(
                f"{type(self).__name__} requires 'class_mapping' to be provided explicitly. "
                "Set 'class_mapping' in TaskConfig or pass it to the encoder constructor."
            )

    def _check_known(self, labels: Iterable[str]) -> None:
        unknown = set(labels) - set(self._label_to_index)
        if unknown:
            raise ValueError(
                f"Column contains labels not in class_mapping: {sorted(unknown)}. "
                f"Known: {sorted(self._label_to_index)}."
            )

    def _index_of(self, label: str) -> int:
        try:
            return self._label_to_index[label]
        except KeyError as error:
            raise KeyError(f"Unknown label {label!r}. Known labels: {sorted(self._label_to_index)}.") from error

    @property
    def num_classes(self) -> int | None:
        return len(self._index_to_label) or None

    @property
    def class_mapping(self) -> dict[int, str]:
        return dict(self._index_to_label)

    def _distribution_from_counts(self, counts: Mapping[str, int]) -> CategoricalDistribution:
        """Order raw label counts by class index, filling absent classes with zero."""
        return CategoricalDistribution(
            counts={label: int(counts.get(label, 0)) for label in self._index_to_label.values()}
        )


@target_encoders.register("label")
class LabelEncoder(_CategoricalEncoder):
    """Maps a single categorical label to its integer class index (multiclass/binary).

    ``load`` resolves the label to its index, so it rides through the transform as an int
    — letting a label-aware augmentation update it (e.g. a rotation aug: ``(index + k) % 4``).
    """

    def fit(self, values: Iterable[Any]) -> None:
        self._require_mapping()
        self._check_known(str(value) for value in values)

    def load(self, value: Any) -> int:
        return self._index_of(str(value))

    def to_tensor(self, value: Any) -> Tensor:
        return torch.tensor(int(value), dtype=torch.long)

    def summarize(self, values: Iterable[Any]) -> Distribution:
        return self._distribution_from_counts(Counter(str(value) for value in values))


@target_encoders.register("multilabel")
class MultiLabelEncoder(_CategoricalEncoder):
    """Maps a delimited label string to a multi-hot ``[C]`` float tensor.

    Parameters:
        separator (str): Delimiter used to split the label string (default ``","``).
        class_mapping (dict[int, str] | None): Fixed vocabulary; ``fit`` only validates.
    """

    def __init__(self, separator: str = ",", class_mapping: dict[int, str] | None = None) -> None:
        super().__init__(class_mapping)
        self._separator = separator

    def _split(self, value: Any) -> list[str]:
        return [part.strip() for part in str(value).split(self._separator) if part.strip()]

    def fit(self, values: Iterable[Any]) -> None:
        self._require_mapping()
        self._check_known(label for value in values for label in self._split(value))

    def load(self, value: Any) -> np.ndarray:
        multi_hot = np.zeros(len(self._index_to_label), dtype=np.float32)
        for label in self._split(value):
            multi_hot[self._index_of(label)] = 1.0
        return multi_hot

    def to_tensor(self, value: Any) -> Tensor:
        return torch.as_tensor(value, dtype=torch.float)

    def summarize(self, values: Iterable[Any]) -> Distribution:
        # Each row contributes one count per active label (multi-hot), so the per-class
        # counts can sum to more than the number of rows.
        return self._distribution_from_counts(Counter(label for value in values for label in self._split(value)))


@target_encoders.register("scalar")
class ScalarEncoder(TargetEncoder):
    """Encodes a scalar numeric target as a ``[]`` float tensor (regression)."""

    def fit(self, values: Iterable[Any]) -> None:
        pass

    def load(self, value: Any) -> float:
        return float(value)

    def to_tensor(self, value: Any) -> Tensor:
        return torch.tensor(float(value), dtype=torch.float)

    def summarize(self, values: Iterable[Any]) -> Distribution | None:
        numbers = np.asarray([float(value) for value in values], dtype=float)
        numbers = numbers[~np.isnan(numbers)]
        if numbers.size == 0:
            return None
        minimum, q25, median, q75, maximum = (float(x) for x in np.percentile(numbers, [0, 25, 50, 75, 100]))
        bin_counts, bin_edges = np.histogram(numbers, bins=min(_REGRESSION_HISTOGRAM_BINS, numbers.size))
        return ContinuousDistribution(
            count=int(numbers.size),
            mean=float(numbers.mean()),
            std=float(numbers.std(ddof=1)) if numbers.size > 1 else 0.0,
            minimum=minimum,
            q25=q25,
            median=median,
            q75=q75,
            maximum=maximum,
            histogram=Histogram(
                counts=tuple(int(count) for count in bin_counts),
                edges=tuple(float(edge) for edge in bin_edges),
            ),
        )


@target_encoders.register("mask")
class MaskEncoder(TargetEncoder):
    """File-based, spatial encoder for index masks: a single-channel PNG of class indices.

    ``load`` reads the PNG (``file_based`` → resolved against ``root_path``) into a ``[H, W]``
    uint8 array before the transform so Albumentations can resize/flip it together with the
    image (``spatial`` → registered as a ``mask`` target). ``to_tensor`` casts the result to a
    ``[H, W]`` long tensor for the criterion.

    When ``class_mapping`` is provided, ``num_classes`` is inferred from it —
    consistent with the categorical encoders. Otherwise ``num_classes`` stays ``None``
    and the task config must supply an explicit ``num_classes``.

    Parameters:
        class_mapping (dict[int, str] | None): Index → label map, e.g.
            ``{0: "background", 1: "defect"}``. Determines class count.
    """

    file_based = True
    spatial = True

    def __init__(self, class_mapping: dict[int, str] | None = None) -> None:
        self._num_classes: int | None = len(class_mapping) if class_mapping is not None else None

    @property
    def num_classes(self) -> int | None:
        return self._num_classes

    def fit(self, values: Iterable[Any]) -> None:
        pass

    def load(self, value: Any) -> np.ndarray:
        mask = cv2.imread(str(value), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found or unreadable: {value}")
        return mask

    def to_tensor(self, value: Any) -> Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.from_numpy(np.asarray(value))
        return tensor.long()


@target_encoders.register("null")
class NullTargetEncoder(TargetEncoder):
    """Placeholder encoder for target-less tasks — needs no annotation column (Null Object).

    Metric-learning tasks supervised purely by batch / triplet structure (triplet,
    contrastive) carry no per-sample label, yet the training step still indexes
    ``batch.targets[task]``. This encoder satisfies that contract without a data column:
    ``load`` ignores its input and ``to_tensor`` yields a ``[]`` zero scalar, which the
    metric criteria ignore. Pair it with ``TargetBinding(column=None)`` (the wiring does
    this automatically when a task config omits ``target``).
    """

    def fit(self, values: Iterable[Any]) -> None:
        pass

    def load(self, value: Any) -> float:
        return 0.0

    def to_tensor(self, value: Any) -> Tensor:
        return torch.tensor(0.0, dtype=torch.float)
