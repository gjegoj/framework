"""One prediction path over ready objects, independent of configuration and Trainer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from src.core import Batch, Prediction, Sample
from src.data import Preprocessor
from src.models import Model
from src.tasks import Task


class Predictor(ABC):
    """Inference components are restored by build.load_predictor, never by this runtime class."""

    def __init__(self, model: Model, tasks: Mapping[str, Task], preprocessor: Preprocessor) -> None:
        self.model = model
        self.tasks = dict(tasks)
        self.preprocessor = preprocessor

    def predict(self, samples: Sequence[Sample]) -> Prediction:
        return self.predict_batch(self._prepare(samples))

    @abstractmethod
    def predict_batch(self, batch: Batch) -> Prediction:
        """Eval/inference mode and shared Task.postprocess; targets are optional."""
        raise NotImplementedError

    def _prepare(self, samples: Sequence[Sample]) -> Batch:
        return self.preprocessor.collate([self.preprocessor.preprocess(sample) for sample in samples])
