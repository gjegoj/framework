"""The standard preprocessor: named encoders per input and target, a transform slot, one pixel finish, one collator."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import batched

from src.console import track
from src.core import Batch, DatasetInfo, Distribution, Geometry, Role, Sample
from src.data.base import Collator, Encoder, InputEncoder, Preprocessor, TargetEncoder
from src.data.cache import Cache, Key
from src.data.registry import preprocessor_registry
from src.transforms import SampleTransform

log = logging.getLogger(__name__)


@preprocessor_registry.register("standard")
class StandardPreprocessor(Preprocessor):
    """load → transform → encode, for every input, target and auxiliary input of a sample.

    ``load`` reads raw cells (through the cache, keyed by what each encoder says identifies the file);
    the stage's transform — the pipeline its YAML declares — moves every pixel-bound value together and
    ends by crossing into tensors; ``encode`` settles each value's training form. The transform is a
    parameter rather than state, because one preprocessor serves every stage and inference: each is
    handed its own pipeline. Auxiliary inputs exist for the transform alone: loaded, never encoded,
    dropped afterwards. A target the sample does not carry is skipped, so the same object predicts
    without labels.
    """

    def __init__(
        self,
        inputs: Mapping[str, InputEncoder],
        targets: Mapping[str, TargetEncoder],
        collator: Collator,
        auxiliary_inputs: Mapping[str, Encoder] | None = None,
        cache: Cache | None = None,
    ) -> None:
        self.inputs = dict(inputs)
        self.targets = dict(targets)
        self.auxiliary_inputs = dict(auxiliary_inputs or {})
        self.collator = collator
        self.cache = cache
        self._by_role: dict[Role, Mapping[str, Encoder]] = {
            Role.INPUTS: self.inputs,
            Role.TARGETS: self.targets,
            Role.AUXILIARY: self.auxiliary_inputs,
        }

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            inputs={name: encoder.info for name, encoder in self.inputs.items()},
            targets={name: encoder.info for name, encoder in self.targets.items()},
        )

    @property
    def geometries(self) -> dict[str, dict[str, Geometry]]:
        """How each declared value moves with the picture, including the ones that do not move at all.

        Everything is published, ``NONE`` included, because what a pipeline can carry is the pipeline's
        to decide: it is the one that knows both — and an augmentation that writes an answer needs a
        value that does not move to reach it.
        """
        return {role: _geometries(encoders) for role, encoders in self._by_role.items()}

    def preprocess(self, sample: Sample, transform: SampleTransform | None = None) -> Sample:
        loaded = Sample(
            inputs={
                name: self._load(Role.INPUTS, name, _required(sample.inputs, name, "input")) for name in self.inputs
            },
            targets={
                name: self._load(Role.TARGETS, name, sample.targets[name])
                for name in self.targets
                if name in sample.targets
            },
            auxiliary_inputs={
                name: self._load(Role.AUXILIARY, name, sample.auxiliary_inputs[name])
                for name in self.auxiliary_inputs
                if name in sample.auxiliary_inputs
            },
            metadata=sample.metadata,
        )
        if transform is not None:
            loaded = transform(loaded)
        return Sample(
            inputs={name: self.inputs[name].encode(value) for name, value in loaded.inputs.items()},
            targets={name: self.targets[name].encode(value) for name, value in loaded.targets.items()},
            metadata=loaded.metadata,
        )

    def collate(self, samples: Sequence[Sample]) -> Batch:
        return self.collator(samples)

    def fit(self, targets: Mapping[str, Iterable[object]]) -> None:
        for name, values in targets.items():
            self.targets[name].fit(values)

    def validate(self, targets: Mapping[str, Iterable[object]]) -> None:
        for name, values in targets.items():
            self.targets[name].validate(values)

    def describe(self, targets: Mapping[str, Iterable[object]]) -> dict[str, Distribution]:
        """Every target whose encoder can say what its cells hold; the rest are simply absent."""
        described = ((name, self.targets[name].distribution(values)) for name, values in targets.items())
        return {name: found for name, found in described if found is not None}

    def warm(self, samples: Iterable[Sample], label: str) -> None:
        """Read every cacheable file of these samples once, a bounded batch of reads at a time."""
        cache = self.cache
        if cache is None:
            return
        pending: dict[Key, tuple[Role, str, object]] = {}
        for sample in samples:
            for role, cells in (
                (Role.INPUTS, sample.inputs),
                (Role.TARGETS, sample.targets),
                (Role.AUXILIARY, sample.auxiliary_inputs),
            ):
                for name, cell in cells.items():
                    key = self._key(role, name, cell)
                    if key is not None and key not in cache and key not in pending:
                        pending[key] = (role, name, cell)
        with cache.filling(), ThreadPoolExecutor(max_workers=cache.workers) as pool:
            reads = (
                done
                for chunk in batched(pending.values(), cache.workers * 8)
                for done in pool.map(lambda task: self._read_quietly(*task), chunk)
            )
            for _ in track(reads, f"Caching {label}", total=len(pending), status=cache.status):
                pass
        log.info("%s: %s", label, cache.summary())

    def _key(self, role: Role, name: str, cell: object) -> Key | None:
        encoder = self._by_role[role].get(name)
        identity = encoder.cache_key(cell) if encoder is not None else None
        return None if identity is None else (role, name, identity)

    def _load(self, role: Role, name: str, cell: object) -> object:
        encoder = self._by_role[role][name]
        if self.cache is None or (key := self._key(role, name, cell)) is None:
            return encoder.load(cell)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        loaded = encoder.load(cell)
        self.cache.put(key, loaded)
        return loaded

    def _read_quietly(self, role: Role, name: str, cell: object) -> None:
        try:
            self._load(role, name, cell)
        except Exception as error:
            log.warning("Cache skipped %s %r (%r): %s", role, name, cell, error)


def _required(values: Mapping[str, object], name: str, role: str) -> object:
    try:
        return values[name]
    except KeyError:
        raise KeyError(f"The sample carries no {role} {name!r}; it has {sorted(values)}.") from None


def _geometries(encoders: Mapping[str, Encoder]) -> dict[str, Geometry]:
    return {name: encoder.geometry for name, encoder in encoders.items()}
