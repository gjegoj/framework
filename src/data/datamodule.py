"""DataModule: framework-agnostic orchestration of the data pipeline.

Deliberately not a ``LightningDataModule`` — it is plain and unit-testable. The
thin Lightning wrapper that delegates here is added in the training layer.

On ``setup`` it: reads the source, fits target codecs (inferring ``num_classes``
into the RuntimeContext), splits into per-stage datasets, and records dataset
sizes. It then hands out standard PyTorch DataLoaders.
"""

from __future__ import annotations

from collections.abc import Mapping

from torch.utils.data import DataLoader

from src.core.enums import Stage
from src.core.runtime import RuntimeContext
from src.data.collate import collate_samples
from src.data.dataset import Dataset, TargetBinding
from src.data.loaders import ImageLoader
from src.data.sources import DataSource
from src.data.split import split_dataframe
from src.data.transforms import Transform


class DataModule:
    """Builds datasets and dataloaders, and populates the RuntimeContext.

    Parameters:
        source (DataSource): Reads the full annotation table.
        bindings (list[TargetBinding]): Per-task target column + codec.
        image_column (str): Column holding image paths.
        transforms (dict[Stage, Transform]): Per-stage input transforms.
        split (dict[Stage, float]): Split ratios.
        runtime (RuntimeContext): Populated with num_classes and dataset sizes.
        batch_size (int): Batch size for all stages.
        seed (int): Split seed.
        num_workers (int): DataLoader worker processes.
        root_path (str | None): Optional image path prefix.
        loader (ImageLoader | None): Image loader (defaults to local file loader).
    """

    def __init__(
        self,
        source: DataSource,
        bindings: list[TargetBinding],
        image_column: str,
        transforms: Mapping[Stage, Transform],
        split: dict[Stage, float],
        runtime: RuntimeContext,
        batch_size: int,
        seed: int,
        num_workers: int = 0,
        root_path: str | None = None,
        loader: ImageLoader | None = None,
    ) -> None:
        self._source = source
        self._bindings = bindings
        self._image_column = image_column
        self._transforms = transforms
        self._split = split
        self._runtime = runtime
        self._batch_size = batch_size
        self._seed = seed
        self._num_workers = num_workers
        self._root_path = root_path
        self._loader = loader or ImageLoader()
        self._datasets: dict[Stage, Dataset] = {}

    def setup(self) -> None:
        """Read data, fit codecs, split, and build per-stage datasets."""
        frame = self._source.read()

        for binding in self._bindings:
            binding.codec.fit(frame[binding.column])
            num_classes = binding.codec.num_classes
            if num_classes is not None:
                self._runtime.num_classes[binding.name] = num_classes

        splits = split_dataframe(frame, self._split, self._seed)
        for stage, part in splits.items():
            self._datasets[stage] = Dataset(
                frame=part,
                image_column=self._image_column,
                bindings=self._bindings,
                transform=self._transforms[stage],
                loader=self._loader,
                root_path=self._root_path,
            )
            self._runtime.dataset_sizes[stage] = len(self._datasets[stage])

    def _dataloader(self, stage: Stage, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            self._datasets[stage],
            batch_size=self._batch_size,
            shuffle=shuffle,
            num_workers=self._num_workers,
            collate_fn=collate_samples,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(Stage.TRAIN, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(Stage.VAL, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        stage = Stage.TEST if Stage.TEST in self._datasets else Stage.VAL
        return self._dataloader(stage, shuffle=False)
