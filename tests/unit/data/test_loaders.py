"""Input loaders: embedding-vector loading and extension-based loader inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import (
    EmbeddingLoader,
)


class TestEmbeddingLoader:
    def test_load_reads_vector_as_float32(self, tmp_path: Path) -> None:
        path = tmp_path / "vec.npy"
        np.save(path, np.arange(8, dtype=np.float64))
        vector = EmbeddingLoader().load(str(path))
        assert vector.shape == (8,)
        assert vector.dtype == np.float32

    def test_load_non_1d_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "matrix.npy"
        np.save(path, np.zeros((4, 8), dtype=np.float32))
        with pytest.raises(ValueError, match="1-D"):
            EmbeddingLoader().load(str(path))

    def test_is_file_based(self) -> None:
        assert EmbeddingLoader().file_based is True


class TestInferLoaderKey:
    def test_npy_paths_infer_embedding(self) -> None:
        from src.data.loaders import infer_loader_key

        series = pd.Series(["emb/a.npy", "emb/b.npy"])
        assert infer_loader_key(series) == "embedding"

    def test_image_paths_still_infer_image(self) -> None:
        from src.data.loaders import infer_loader_key

        series = pd.Series(["img/a.jpg", "img/b.png"])
        assert infer_loader_key(series) == "image"
