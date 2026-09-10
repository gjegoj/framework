"""Encoders whose cells are paths: one rule for where a file lives and what identifies it in a cache."""

from __future__ import annotations

from pathlib import Path

from src.data.base import Encoder


class FileEncoder(Encoder):
    """A cell names a file under ``root``; its resolved path identifies it across roots and runs."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else None

    def path_of(self, value: object) -> Path:
        return self.root / str(value) if self.root is not None else Path(str(value))

    def cache_key(self, value: object) -> str:
        return str(self.path_of(value).resolve())
