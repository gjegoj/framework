"""Lightning DataModule wrapper: a humble adapter over the plain DataModule.

The plain ``DataModule`` is framework-agnostic and fully testable in isolation.
This class adds the ``pl.LightningDataModule`` interface that the Trainer
expects, delegating everything to the inner module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as L
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from src.data.datamodule import DataModule as PlainDataModule


class LitDataModule(L.LightningDataModule):
    """Thin Lightning adapter over :class:`~src.data.datamodule.DataModule`.

    Parameters:
        datamodule (PlainDataModule): The plain, framework-agnostic data module.
    """

    def __init__(self, datamodule: PlainDataModule) -> None:
        super().__init__()
        self._inner = datamodule

    def setup(self, stage: str | None = None) -> None:
        self._inner.setup()

    def train_dataloader(self) -> DataLoader:
        return self._inner.train_dataloader()

    def val_dataloader(self) -> DataLoader:
        return self._inner.val_dataloader()

    def test_dataloader(self) -> DataLoader:
        return self._inner.test_dataloader()
